/**
 * Builds the React UI bundle (src/app) for the browser.
 * Served from memory by src/server.ts — no separate build step, no watcher:
 * dev requests rebuild automatically when a src/app file changes.
 */
import { readdirSync, statSync, type Dirent } from "node:fs";

const APP_DIR = `${import.meta.dir}/app`;
const APP_ENTRY = `${APP_DIR}/main.tsx`;
const APP_TSCONFIG = `${import.meta.dir}/../tsconfig.json`;

export async function buildAppBundle(): Promise<{ js: string; ok: boolean }> {
  try {
    const result = await Bun.build({
      entrypoints: [APP_ENTRY],
      target: "browser",
      format: "iife",
      minify: process.env.NODE_ENV === "production",
      define: {
        "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV || "development"),
      },
      tsconfig: APP_TSCONFIG,
    });
    if (!result.success) {
      console.error("[app-build] bundle failed:");
      for (const log of result.logs) console.error(String(log));
      return { js: "", ok: false };
    }
    const out = result.outputs[0];
    if (!out) return { js: "", ok: false };
    return { js: await out.text(), ok: true };
  } catch (e) {
    console.error("[app-build] unexpected failure:", e);
    return { js: "", ok: false };
  }
}

/** Newest mtime (ms) under the app source tree — used for dev rebuild-on-request. */
export function latestAppMtime(): number {
  const walk = (dir: string): number => {
    let latest = 0;
    let entries: Dirent[] = [];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch (_) {
      return 0;
    }
    for (const e of entries) {
      const p = `${dir}/${e.name}`;
      if (e.isDirectory()) latest = Math.max(latest, walk(p));
      else if (e.isFile()) {
        try {
          latest = Math.max(latest, statSync(p).mtimeMs);
        } catch (_) {}
      }
    }
    return latest;
  };
  return walk(APP_DIR);
}
