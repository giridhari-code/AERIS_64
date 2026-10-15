/**
 * Safe git helpers for the Bun edge layer.
 * Mirrors the Python git_tool: status / log / diff only by default.
 */

import { $ } from "bun";

const REPO_ROOT =
  process.env.AERIS_REPO_ROOT ||
  process.env.GITHUB_WORKSPACE ||
  process.cwd();

async function runGit(args: string[]): Promise<{
  ok: boolean;
  output: string;
  error?: string;
}> {
  try {
    const proc = Bun.spawn(["git", "-C", REPO_ROOT, ...args], {
      stdout: "pipe",
      stderr: "pipe",
    });
    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const code = await proc.exited;
    if (code !== 0) {
      return {
        ok: false,
        output: stdout.trim(),
        error: stderr.trim() || `exit ${code}`,
      };
    }
    return { ok: true, output: (stdout || stderr).trim() };
  } catch (e) {
    return {
      ok: false,
      output: "",
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

export async function gitStatus() {
  const r = await runGit(["status", "--short", "--branch"]);
  return { tool: "git_status", repo: REPO_ROOT, ...r };
}

export async function gitLog() {
  const r = await runGit(["log", "--oneline", "-20"]);
  return { tool: "git_log", repo: REPO_ROOT, ...r };
}

export async function gitDiff() {
  const r = await runGit(["diff", "--stat"]);
  return { tool: "git_diff", repo: REPO_ROOT, ...r };
}
