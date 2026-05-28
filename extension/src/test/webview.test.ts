// Runs the REAL bundled dist/webview/main.js (not a reimplementation of
// its logic) inside a simulated DOM via jsdom -- this is the part that
// can't be verified any other way in this environment: there's no way to
// drive an actual VS Code webview from here. It caught two real bugs
// already found by hand (Markdown syntax showing as literal asterisks;
// the panel wiping the previous answer on every new question) --
// this test exists so neither regresses silently.
//
// Requires `npm run compile` to have produced dist/webview/main.js first.
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const bundlePath = path.join(here, "..", "..", "dist", "webview", "main.js");

function setUpWebview() {
  const dom = new JSDOM(
    `<!DOCTYPE html><body>
      <div id="transcript"></div>
      <form id="ask-form">
        <textarea id="question"></textarea>
        <button type="submit">Ask</button>
      </form>
    </body>`,
    { runScripts: "outside-only" },
  );
  const posted: unknown[] = [];
  // jsdom's window is close enough to the webview's real global scope for
  // this bundle's needs (DOM + postMessage), not a full browser.
  (dom.window as unknown as { acquireVsCodeApi: () => unknown }).acquireVsCodeApi = () => ({
    postMessage: (msg: unknown) => posted.push(msg),
  });
  // jsdom doesn't implement layout, so scrollIntoView isn't there --
  // a real gap in the test environment, not something the bundle itself
  // is missing, so it's stubbed rather than removed from main.js.
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};

  const bundle = readFileSync(bundlePath, "utf-8");
  dom.window.eval(bundle);

  return { dom, posted };
}

let bundleExists = true;
before(() => {
  try {
    readFileSync(bundlePath);
  } catch {
    bundleExists = false;
  }
});

test("webview: markdown renders to real elements, not literal syntax", () => {
  if (!bundleExists) {
    throw new Error("dist/webview/main.js not found -- run `npm run compile` first");
  }
  const { dom } = setUpWebview();
  const { document, Event } = dom.window;

  const textarea = document.getElementById("question") as InstanceType<typeof dom.window.HTMLTextAreaElement>;
  textarea.value = "what does escape do?";
  document.getElementById("ask-form")!.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));

  dom.window.dispatchEvent(
    new dom.window.MessageEvent("message", {
      data: { type: "token", text: "It **escapes** HTML in `src/app.py:10-20`." },
    }),
  );

  const answerEl = document.querySelector(".turn-answer")!;
  assert.equal(answerEl.querySelector("strong")?.textContent, "escapes");
  const citation = answerEl.querySelector(".citation");
  assert.ok(citation, "citation span should exist even though it was inside backticks");
  assert.equal(citation!.getAttribute("data-citation"), "src/app.py:10-20");
  // The literal markdown syntax must not survive into the rendered text.
  assert.ok(!answerEl.textContent!.includes("**"));
});

test("webview: a second question appends a new turn instead of replacing the first", () => {
  if (!bundleExists) {
    throw new Error("dist/webview/main.js not found -- run `npm run compile` first");
  }
  const { dom } = setUpWebview();
  const { document, Event } = dom.window;

  const textarea = document.getElementById("question") as InstanceType<typeof dom.window.HTMLTextAreaElement>;
  const form = document.getElementById("ask-form")!;

  textarea.value = "question one";
  form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
  dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data: { type: "token", text: "answer one" } }));

  textarea.value = "question two";
  form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
  dom.window.dispatchEvent(new dom.window.MessageEvent("message", { data: { type: "token", text: "answer two" } }));

  const turns = document.querySelectorAll(".turn");
  assert.equal(turns.length, 2, "both turns should still be present in the transcript");
  assert.ok(turns[0].textContent!.includes("question one"));
  assert.ok(turns[0].textContent!.includes("answer one"));
  assert.ok(turns[1].textContent!.includes("question two"));
  assert.ok(turns[1].textContent!.includes("answer two"));
});
