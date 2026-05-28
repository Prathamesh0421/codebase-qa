// Two separate bundles, because the extension host and the webview run in
// two fundamentally different JS contexts and neither can import the
// other's code: dist/extension.js (Node, CommonJS, the "vscode" module
// available as a host global) and dist/webview/main.js (browser/IIFE, no
// Node APIs, bundles its own dependencies like `marked` since a webview
// can't reach node_modules at runtime -- CSP blocks any script that isn't
// the extension's own bundled file). styles.css is a plain static asset,
// just copied into place.
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const production = process.argv.includes("--production");
const watch = process.argv.includes("--watch");

function copyStyles() {
  const destDir = path.join(__dirname, "dist", "webview");
  fs.mkdirSync(destDir, { recursive: true });
  fs.copyFileSync(
    path.join(__dirname, "src", "webview", "styles.css"),
    path.join(destDir, "styles.css"),
  );
}

async function main() {
  copyStyles();

  const extensionCtx = await esbuild.context({
    entryPoints: ["src/extension.ts"],
    bundle: true,
    format: "cjs",
    platform: "node",
    target: "node18",
    outfile: "dist/extension.js",
    external: ["vscode"],
    sourcemap: !production,
    minify: production,
    logLevel: "info",
  });

  const webviewCtx = await esbuild.context({
    entryPoints: ["src/webview/main.js"],
    bundle: true,
    format: "iife",
    platform: "browser",
    target: "es2022",
    outfile: "dist/webview/main.js",
    sourcemap: !production,
    minify: production,
    logLevel: "info",
    plugins: [
      {
        name: "copy-styles-on-rebuild",
        setup(build) {
          build.onEnd(copyStyles);
        },
      },
    ],
  });

  if (watch) {
    await Promise.all([extensionCtx.watch(), webviewCtx.watch()]);
  } else {
    await Promise.all([extensionCtx.rebuild(), webviewCtx.rebuild()]);
    await Promise.all([extensionCtx.dispose(), webviewCtx.dispose()]);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
