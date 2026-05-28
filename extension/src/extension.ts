import * as vscode from "vscode";
import { CodeQAPanel } from "./panel";
import { promptForApiKey } from "./secrets";

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("codeqa.ask", () => {
      CodeQAPanel.createOrShow(context);
    }),
    vscode.commands.registerCommand("codeqa.setApiKey", async () => {
      const key = await promptForApiKey(context);
      if (key) {
        void vscode.window.showInformationMessage("CodeQA API key saved.");
      }
    }),
  );
}

export function deactivate(): void {
  // Nothing to tear down explicitly -- the panel's own onDidDispose
  // handler cancels any in-flight request, and VS Code disposes
  // everything in context.subscriptions on unload.
}
