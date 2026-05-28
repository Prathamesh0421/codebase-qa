import * as vscode from "vscode";
import { streamQuery, QueryHttpError } from "./sseClient";
import { getApiKey, promptForApiKey } from "./secrets";

// Citation format is exactly RetrievedChunk.citation's "path:start-end" --
// verified against grounding.py's own regex (word chars, dots, slashes,
// hyphens for the path; digits for the line numbers).
const CITATION_RE = /([\w./-]+):(\d+)-(\d+)/;

type WebviewMessage =
  | { type: "ask"; question: string }
  | { type: "openCitation"; citation: string };

export class CodeQAPanel {
  private static current: CodeQAPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly context: vscode.ExtensionContext;
  private inFlight: AbortController | undefined;
  private readonly disposables: vscode.Disposable[] = [];

  static createOrShow(context: vscode.ExtensionContext): void {
    if (CodeQAPanel.current) {
      CodeQAPanel.current.panel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel("codeqaAsk", "CodeQA", vscode.ViewColumn.Beside, {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, "dist", "webview")],
      retainContextWhenHidden: true,
    });
    CodeQAPanel.current = new CodeQAPanel(panel, context);
  }

  private constructor(panel: vscode.WebviewPanel, context: vscode.ExtensionContext) {
    this.panel = panel;
    this.context = context;
    this.panel.webview.html = this.buildHtml();

    this.panel.webview.onDidReceiveMessage(
      (message: WebviewMessage) => void this.handleMessage(message),
      undefined,
      this.disposables,
    );
    this.panel.onDidDispose(() => this.dispose(), undefined, this.disposables);
  }

  private async handleMessage(message: WebviewMessage): Promise<void> {
    if (message.type === "ask") {
      await this.ask(message.question);
    } else if (message.type === "openCitation") {
      await this.openCitation(message.citation);
    }
  }

  private async ask(question: string): Promise<void> {
    // A new question cancels whatever's still streaming from a previous
    // one -- otherwise two answers could interleave tokens in the panel.
    this.inFlight?.abort();
    const controller = new AbortController();
    this.inFlight = controller;

    const config = vscode.workspace.getConfiguration("codeqa");
    const backendUrl = config.get<string>("backendUrl", "");
    const repoSlug = config.get<string>("defaultRepo", "");

    if (!backendUrl || !repoSlug) {
      this.post({
        type: "error",
        message: "Set codeqa.backendUrl and codeqa.defaultRepo in Settings first.",
      });
      return;
    }

    let apiKey = await getApiKey(this.context);
    if (!apiKey) {
      apiKey = await promptForApiKey(this.context);
    }
    if (!apiKey) {
      this.post({ type: "error", message: "CodeQA API key is required (run CodeQA: Set API Key)." });
      return;
    }

    try {
      for await (const event of streamQuery(backendUrl, apiKey, repoSlug, question, controller.signal)) {
        this.post(event);
      }
    } catch (err) {
      if (controller.signal.aborted) {
        return; // superseded by a newer question -- not a real error
      }
      if (err instanceof QueryHttpError && err.status === 401) {
        this.post({ type: "error", message: "Invalid or revoked API key. Run CodeQA: Set API Key." });
      } else if (err instanceof QueryHttpError) {
        this.post({ type: "error", message: `${err.status}: ${err.message}` });
      } else {
        this.post({ type: "error", message: String(err) });
      }
    }
  }

  private async openCitation(citation: string): Promise<void> {
    const match = CITATION_RE.exec(citation);
    if (!match) {
      return;
    }
    const [, path, startStr, endStr] = match;
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) {
      void vscode.window.showWarningMessage("CodeQA: no workspace folder open to resolve the citation against.");
      return;
    }
    const start = parseInt(startStr, 10) - 1; // citations are 1-indexed; VS Code ranges are 0-indexed
    const end = parseInt(endStr, 10) - 1;
    const uri = vscode.Uri.joinPath(folder.uri, path);
    try {
      const document = await vscode.workspace.openTextDocument(uri);
      const selection = new vscode.Range(start, 0, end, document.lineAt(Math.min(end, document.lineCount - 1)).text.length);
      await vscode.window.showTextDocument(document, { selection, viewColumn: vscode.ViewColumn.One });
    } catch {
      void vscode.window.showWarningMessage(`CodeQA: couldn't open ${path} in the current workspace.`);
    }
  }

  private post(event: unknown): void {
    void this.panel.webview.postMessage(event);
  }

  private buildHtml(): string {
    const webview = this.panel.webview;
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "dist", "webview", "main.js"),
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "dist", "webview", "styles.css"),
    );
    const nonce = getNonce();

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
  <link href="${styleUri}" rel="stylesheet" />
  <title>CodeQA</title>
</head>
<body>
  <div id="app">
    <div id="transcript" role="log" aria-live="polite"></div>
    <form id="ask-form">
      <textarea id="question" rows="2" placeholder="Ask about this codebase..."></textarea>
      <button type="submit">Ask</button>
    </form>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  private dispose(): void {
    this.inFlight?.abort();
    CodeQAPanel.current = undefined;
    for (const d of this.disposables) {
      d.dispose();
    }
    this.panel.dispose();
  }
}

function getNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
