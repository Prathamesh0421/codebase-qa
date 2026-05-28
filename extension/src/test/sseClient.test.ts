import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBlock, toQueryEvent, streamQuery, QueryHttpError } from "../sseClient.ts";

test("parseBlock: single event/data line", () => {
  const result = parseBlock("event: token\ndata: hello");
  assert.deepEqual(result, { event: "token", data: "hello" });
});

test("parseBlock: multiple data: lines join with newline", () => {
  // Verified against the real backend: a single token event can arrive
  // with its text split across several data: lines (the SSE spec's way of
  // preserving embedded newlines in one logical event).
  const result = parseBlock("event: token\ndata: line one\ndata: line two");
  assert.deepEqual(result, { event: "token", data: "line one\nline two" });
});

test("parseBlock: a block with only a comment line is not an event", () => {
  // sse-starlette's keepalive pings look like ": ping - 2026-01-01..." --
  // no event: or data: line at all.
  const result = parseBlock(": ping - 2026-08-19 00:00:00");
  assert.equal(result, null);
});

test("parseBlock: missing event: line defaults to 'message'", () => {
  const result = parseBlock("data: bare data, no event name");
  assert.deepEqual(result, { event: "message", data: "bare data, no event name" });
});

test("toQueryEvent: token data is used as raw text, not JSON-parsed", () => {
  // Verified against the real backend: synthesize()'s streamed tokens go
  // out as raw strings via a custom writer channel, never JSON-encoded --
  // unlike every other event type.
  const event = toQueryEvent("token", "Based on `app.py:1-5`");
  assert.deepEqual(event, { type: "token", text: "Based on `app.py:1-5`" });
});

test("toQueryEvent: progress data is JSON and spread onto the event", () => {
  const event = toQueryEvent("progress", JSON.stringify({ stage: "locate", attempt: 1, chunk_count: 30 }));
  assert.deepEqual(event, { type: "progress", stage: "locate", attempt: 1, chunk_count: 30 });
});

test("toQueryEvent: done maps citations_dropped (snake_case wire format) to citationsDropped", () => {
  const event = toQueryEvent(
    "done",
    JSON.stringify({ chunks: [{ citation: "a.py:1-2" }], citations_dropped: [] }),
  );
  assert.deepEqual(event, {
    type: "done",
    chunks: [{ citation: "a.py:1-2" }],
    citationsDropped: [],
  });
});

test("toQueryEvent: an unrecognized event name becomes a visible error, not a silent drop", () => {
  const event = toQueryEvent("mystery", "{}");
  assert.equal(event.type, "error");
});

function sseResponse(body: string, status = 200): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(body));
      controller.close();
    },
  });
  return new Response(status === 200 ? stream : null, { status });
}

test("streamQuery: yields events in order and skips ping comments", async (t) => {
  const body = [
    'event: progress\ndata: {"stage": "locate", "attempt": 1, "chunk_count": 5}',
    ": ping - 2026-08-19 00:00:00",
    'event: token\ndata: hello',
    'event: token\ndata:  world',
    'event: done\ndata: {"chunks": [], "citations_dropped": []}',
  ].join("\n\n") + "\n\n";

  t.mock.method(global, "fetch", async () => sseResponse(body));

  const events = [];
  for await (const event of streamQuery("https://example.test", "key", "repo", "q", new AbortController().signal)) {
    events.push(event);
  }

  assert.deepEqual(events, [
    { type: "progress", stage: "locate", attempt: 1, chunk_count: 5 },
    { type: "token", text: "hello" },
    { type: "token", text: " world" },
    { type: "done", chunks: [], citationsDropped: [] },
  ]);
});

function sseResponseFromChunks(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(status === 200 ? stream : null, { status });
}

test("streamQuery: handles real CRLF line endings delivered across multiple network chunks", async (t) => {
  // Reproduces exactly what the live backend actually sends -- \r\n line
  // endings, and one logical event split across several separate
  // reader.read() chunks with a citation broken mid-line. A plain-\n,
  // single-chunk mock (the test above) passed while the real client
  // silently yielded zero events against production, because \r\n\r\n
  // never matches a \n\n block separator. This is the regression test for
  // that bug.
  const chunks = [
    'event: progress\r\ndata: {"stage": "locate", "attempt": 1, "chunk_count": 17}\r\n\r\n',
    ": ping - 2026-08-19 02:45:04.596822+00:00\r\n\r\n",
    'event: progress\r\ndata: {"stage": "trace", "sufficient": true}\r\n\r\n',
    'event: token\r\ndata: `escape_silent` is a function that escapes characters (`src/mark\r\n\r\n',
    'event: token\r\ndata: upsafe/__init__.py:48-56`).\r\n\r\n',
    'event: done\r\ndata: {"chunks": [], "citations_dropped": []}\r\n\r\n',
  ];

  t.mock.method(global, "fetch", async () => sseResponseFromChunks(chunks));

  const events = [];
  for await (const event of streamQuery("https://example.test", "key", "repo", "q", new AbortController().signal)) {
    events.push(event);
  }

  assert.deepEqual(events, [
    { type: "progress", stage: "locate", attempt: 1, chunk_count: 17 },
    { type: "progress", stage: "trace", sufficient: true },
    { type: "token", text: "`escape_silent` is a function that escapes characters (`src/mark" },
    { type: "token", text: "upsafe/__init__.py:48-56`)." },
    { type: "done", chunks: [], citationsDropped: [] },
  ]);
});

test("streamQuery: a non-200 response raises QueryHttpError with the detail from the JSON body", async (t) => {
  t.mock.method(
    global,
    "fetch",
    async () =>
      new Response(JSON.stringify({ detail: "rate limit exceeded" }), {
        status: 429,
        statusText: "Too Many Requests",
      }),
  );

  await assert.rejects(
    async () => {
      for await (const _ of streamQuery("https://example.test", "key", "repo", "q", new AbortController().signal)) {
        // draining the generator to trigger the throw
      }
    },
    (err: unknown) => {
      assert.ok(err instanceof QueryHttpError);
      assert.equal(err.status, 429);
      assert.equal(err.message, "rate limit exceeded");
      return true;
    },
  );
});
