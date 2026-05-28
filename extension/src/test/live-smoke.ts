// Not part of the automated test suite (not matched by `npm test`'s glob,
// and requires real credentials) -- a one-off script to run the actual
// sseClient.ts code path against the real deployed backend, the same way
// panel.ts does, to prove the SSE parsing logic is correct against real
// production output, not just a hand-built mock.
import { streamQuery } from "../sseClient.ts";

const backendUrl = process.env.CODEQA_BACKEND_URL;
const apiKey = process.env.CODEQA_API_KEY;
const repoSlug = process.env.CODEQA_REPO_SLUG ?? "markupsafe";

if (!backendUrl || !apiKey) {
  console.error("Set CODEQA_BACKEND_URL and CODEQA_API_KEY");
  process.exit(1);
}

const controller = new AbortController();
let tokenCount = 0;
let fullAnswer = "";

for await (const event of streamQuery(backendUrl, apiKey, repoSlug, "what does escape_silent do?", controller.signal)) {
  if (event.type === "progress") {
    console.log(`[progress] ${JSON.stringify(event)}`);
  } else if (event.type === "token") {
    tokenCount++;
    fullAnswer += event.text;
  } else if (event.type === "done") {
    console.log(`[done] ${event.chunks.length} chunks, ${event.citationsDropped.length} dropped citations`);
  } else if (event.type === "error") {
    console.error(`[error] ${event.message}`);
    process.exit(1);
  }
}

console.log(`\nReceived ${tokenCount} token events, ${fullAnswer.length} chars total.\n`);
console.log(fullAnswer);
