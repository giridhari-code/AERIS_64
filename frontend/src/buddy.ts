/**
 * Buddy UI controller — /buddy on|off syncs float + navbar everywhere.
 * Used by the chat page; pure client logic (no model).
 */

export type BuddyState = {
  on: boolean;
  userId: string;
};

const STORAGE_ON = "sentinel_buddy_on";
const STORAGE_UID = "sentinel_buddy_uid";

export function readBuddyOn(): boolean {
  try {
    return localStorage.getItem(STORAGE_ON) === "1";
  } catch {
    return false;
  }
}

export function writeBuddyOn(on: boolean): void {
  try {
    localStorage.setItem(STORAGE_ON, on ? "1" : "0");
  } catch {
    /* ignore */
  }
}

export function buddyUserId(): string {
  try {
    let id = localStorage.getItem(STORAGE_UID);
    if (!id) {
      id = "web-" + Math.random().toString(36).slice(2, 10);
      localStorage.setItem(STORAGE_UID, id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}

/** Parse chat line → on | off | null */
export function parseBuddyCommand(text: string): "on" | "off" | null {
  const m = (text || "").trim().match(/^\/\s*buddy\s*(on|off)?\s*$/i);
  if (!m) return null;
  const sub = (m[1] || "on").toLowerCase();
  return sub === "off" ? "off" : "on";
}

/**
 * Chat-line matcher used by the UI (broader than parseBuddyCommand):
 * accepts /buddy on|off, --buddy on|off, and plain "buddy on|off" (+ karo/kardo).
 */
export function matchBuddyOnOff(text: string): "on" | "off" | null {
  const q = (text || "").trim();
  const m =
    q.match(/^(?:\/|--)\s*buddy\s*(on|off)?\s*$/i) ||
    q.match(/^buddy\s+(on|off)(?:\s+(?:karo|kardo|kar\s*do))?\s*$/i);
  if (!m) return null;
  return (m[1] || "on").toLowerCase() === "off" ? "off" : "on";
}

export function syncBuddyDom(on: boolean): void {
  const btn = document.getElementById("btnBuddy");
  if (btn) {
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.classList.toggle("active", on);
    btn.title = on ? "Buddy on — type /buddy off" : "Buddy off — type /buddy on";
  }
  const panel = document.getElementById("buddy-float");
  if (panel) {
    panel.classList.toggle("open", on);
    panel.setAttribute("aria-hidden", on ? "false" : "true");
    (panel as HTMLElement).style.display = on ? "block" : "none";
  }
  if (!on) {
    const ascii = document.getElementById("buddyAscii");
    const meta = document.getElementById("buddyMeta");
    if (ascii) ascii.textContent = "";
    if (meta) meta.textContent = "";
  }
}
