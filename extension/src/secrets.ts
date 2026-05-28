import * as vscode from "vscode";

// SecretStorage, not a plain setting -- a codeqa.apiKey configuration
// value would sync in plaintext via Settings Sync and show up in
// settings.json. SecretStorage is backed by the OS keychain.
const API_KEY_SECRET = "codeqa.apiKey";

export async function getApiKey(context: vscode.ExtensionContext): Promise<string | undefined> {
  return context.secrets.get(API_KEY_SECRET);
}

export async function setApiKey(context: vscode.ExtensionContext, key: string): Promise<void> {
  await context.secrets.store(API_KEY_SECRET, key);
}

export async function promptForApiKey(context: vscode.ExtensionContext): Promise<string | undefined> {
  const key = await vscode.window.showInputBox({
    title: "CodeQA API Key",
    prompt: "Paste the key from `codeqa keys create` (shown once at creation).",
    password: true,
    ignoreFocusOut: true,
    validateInput: (value) => (value.trim().length > 0 ? undefined : "Key cannot be empty"),
  });
  if (!key) {
    return undefined;
  }
  await setApiKey(context, key.trim());
  return key.trim();
}
