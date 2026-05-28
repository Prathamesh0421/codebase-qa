// Runs inside the sandboxed webview -- no vscode.* API access, no direct
// network access assumed. Everything it knows comes from postMessage
// events the extension host sends; everything it wants done (ask a
// question, open a citation) it asks the host to do via postMessage back.
import { marked } from "marked";

(function () {
  const vscode = acquireVsCodeApi();

  const form = document.getElementById("ask-form");
  const textarea = document.getElementById("question");
  const transcript = document.getElementById("transcript");

  // Citation format is exactly RetrievedChunk.citation's "path:start-end".
  const CITATION_RE = /^([\w./-]+):(\d+)-(\d+)$/;
  // Same pattern, global, for scanning a text node for embedded matches
  // rather than testing a whole string against it.
  const CITATION_SCAN_RE = /([\w./-]+):(\d+)-(\d+)/g;

  // Each element is one question/answer pair, appended and never removed
  // -- a running transcript, not a single area overwritten on every ask
  // (an earlier version wiped the previous answer on each new question;
  // fixed after that was flagged as unexpected against the real thing).
  let currentTurn = null;

  function newTurn(question) {
    const turn = document.createElement("div");
    turn.className = "turn";

    const q = document.createElement("div");
    q.className = "turn-question";
    q.textContent = question;
    turn.appendChild(q);

    const answer = document.createElement("div");
    answer.className = "turn-answer";
    turn.appendChild(answer);

    const status = document.createElement("div");
    status.className = "turn-status";
    turn.appendChild(status);

    transcript.appendChild(turn);
    turn.scrollIntoView({ behavior: "smooth", block: "end" });

    return { root: turn, answerEl: answer, statusEl: status, rawText: "" };
  }

  // Walks every text node under root and replaces citation-pattern
  // substrings with clickable spans. Runs AFTER marked has turned the raw
  // answer into real HTML (bold, headers, code spans, lists) -- operating
  // on rendered text nodes, not the raw markdown source, is what lets a
  // citation still get linkified when the model wraps it in backticks
  // (marked's own inline-code handling would otherwise swallow any
  // markdown link syntax placed inside the backticks before parsing).
  function linkifyCitations(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const targets = [];
    let node;
    while ((node = walker.nextNode())) {
      if (CITATION_SCAN_RE.test(node.nodeValue)) {
        targets.push(node);
      }
      CITATION_SCAN_RE.lastIndex = 0;
    }
    for (const textNode of targets) {
      const parent = textNode.parentNode;
      const fragment = document.createDocumentFragment();
      let lastIndex = 0;
      const text = textNode.nodeValue;
      CITATION_SCAN_RE.lastIndex = 0;
      let match;
      while ((match = CITATION_SCAN_RE.exec(text))) {
        if (match.index > lastIndex) {
          fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
        }
        const span = document.createElement("span");
        span.className = "citation";
        span.dataset.citation = match[0];
        span.textContent = match[0];
        fragment.appendChild(span);
        lastIndex = match.index + match[0].length;
      }
      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
      }
      parent.replaceChild(fragment, textNode);
    }
  }

  function renderTurnAnswer(turn) {
    // marked.parse() on a truncated mid-stream markdown string can render
    // a little rough at the trailing edge (an unclosed ** or list item) --
    // accepted as a minor, self-correcting rough edge: each subsequent
    // token re-parses the whole accumulated text from scratch, so it
    // settles to fully correct HTML by the time the "done" event arrives.
    turn.answerEl.innerHTML = marked.parse(turn.rawText);
    linkifyCitations(turn.answerEl);
  }

  transcript.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.classList.contains("citation")) {
      vscode.postMessage({ type: "openCitation", citation: target.dataset.citation });
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!question) {
      return;
    }
    textarea.value = "";
    currentTurn = newTurn(question);
    currentTurn.statusEl.textContent = "asking...";
    vscode.postMessage({ type: "ask", question });
  });

  window.addEventListener("message", (event) => {
    if (!currentTurn) {
      return;
    }
    const message = event.data;
    switch (message.type) {
      case "progress":
        if (message.stage === "locate") {
          currentTurn.statusEl.textContent = `locate (attempt ${message.attempt}): ${message.chunk_count} chunks so far`;
        } else if (message.stage === "trace") {
          currentTurn.statusEl.textContent = message.sufficient
            ? "trace: sufficient, synthesizing..."
            : "trace: insufficient, refining...";
        }
        break;
      case "token":
        currentTurn.rawText += message.text;
        renderTurnAnswer(currentTurn);
        currentTurn.root.scrollIntoView({ behavior: "smooth", block: "end" });
        break;
      case "done":
        currentTurn.statusEl.textContent =
          message.citationsDropped && message.citationsDropped.length > 0
            ? `${message.citationsDropped.length} citation(s) could not be verified.`
            : "";
        break;
      case "error":
        currentTurn.statusEl.textContent = `error: ${message.message}`;
        break;
    }
  });
})();
