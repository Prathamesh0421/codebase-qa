// Consumes POST /v1/query's SSE stream. Runs in the extension host (Node),
// never in the webview -- the host is what holds the API key and has
// unrestricted network access, matching the "webview is a thin render
// surface" split described in the design.

export type QueryEvent =
  | { type: "progress"; stage: string; [key: string]: unknown }
  | { type: "token"; text: string }
  | { type: "done"; chunks: unknown[]; citationsDropped: unknown[] }
  | { type: "error"; message: string };

export class QueryHttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// Parses one SSE "block" (everything between blank-line separators) into
// its event name and joined data. A block with no "event:" line and no
// "data:" line at all is a bare comment (sse-starlette's `: ping - ...`
// keepalive) and is intentionally not represented here -- callers only
// ever see real events.
export function parseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith(":")) {
      continue; // comment / keepalive
    }
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trimStart();
    } else if (line.startsWith("data:")) {
      // SSE allows multiple data: lines per event, joined by "\n" to
      // reconstruct multi-line text -- observed directly against the real
      // backend, where a single token event can span several data: lines.
      dataLines.push(line.slice("data:".length).replace(/^ /, ""));
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}

export async function* streamQuery(
  backendUrl: string,
  apiKey: string,
  repoSlug: string,
  question: string,
  signal: AbortSignal,
): AsyncGenerator<QueryEvent> {
  const response = await fetch(`${backendUrl.replace(/\/$/, "")}/v1/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ repo_slug: repoSlug, question }),
    signal,
  });

  if (response.status !== 200 || !response.body) {
    // A 429/401/404 is a real HTTP error response, raised by the backend
    // before EventSourceResponse is ever constructed (see api/app.py) --
    // never an SSE "error" event, so it's handled here, not in the parser.
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // body wasn't JSON -- fall back to statusText already set above
    }
    throw new QueryHttpError(response.status, detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return;
      }
      // Real network responses use \r\n line endings (verified against
      // the live backend -- a plain-\n mock is not representative of
      // actual wire output), so \r\n is normalized to \n before anything
      // else. Without this, the \n\n block separator below never matches
      // \r\n\r\n and no event is ever recognized as complete.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      // SSE events are separated by a blank line (\n\n). The last element
      // after split is either "" (buffer ended exactly on a separator) or
      // a partial block still awaiting more bytes -- put it back.
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";

      for (const block of blocks) {
        const parsed = parseBlock(block);
        if (parsed === null) {
          continue;
        }
        yield toQueryEvent(parsed.event, parsed.data);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function toQueryEvent(event: string, data: string): QueryEvent {
  switch (event) {
    case "token":
      // Plain text, not JSON -- verified against the real backend
      // (synthesize()'s streamed tokens go out via a custom writer
      // channel as raw strings, not encoded).
      return { type: "token", text: data };
    case "progress": {
      const payload = JSON.parse(data) as { stage: string; [key: string]: unknown };
      return { type: "progress", ...payload };
    }
    case "done": {
      const payload = JSON.parse(data) as { chunks: unknown[]; citations_dropped: unknown[] };
      return { type: "done", chunks: payload.chunks, citationsDropped: payload.citations_dropped };
    }
    case "error": {
      const payload = JSON.parse(data) as { message: string };
      return { type: "error", message: payload.message };
    }
    default:
      return { type: "error", message: `unrecognized SSE event: ${event}` };
  }
}
