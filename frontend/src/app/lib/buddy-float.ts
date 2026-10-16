/**
 * Buddy float controller — imperative port of the legacy inline script.
 * Same DOM ids (#buddy-float, #buddyAscii, #buddyMeta), same window hooks
 * (sentinelBuddyReact / sentinelBuddySetVisible / sentinelBuddyPet).
 */
import { buddyUserId as storedBuddyUserId, readBuddyOn, writeBuddyOn } from "../../buddy";

const CSS_ANIMS = ["float", "talk", "think", "happy", "sleep", "excited", "wave", "pet"];

interface BuddyData {
  name?: string;
  species_name?: string;
  animations?: Record<string, string[]>;
  sprite_lines?: string[];
}

let inited = false;

export function initBuddy(): void {
  if (inited) return;
  inited = true;

  const buddyFloat = document.getElementById("buddy-float");
  const buddyAscii = document.getElementById("buddyAscii");
  const buddyMeta = document.getElementById("buddyMeta");

  let buddyOn = readBuddyOn();
  let buddyFrames: string[] | null = null;
  let buddyFrameIdx = 0;
  let buddyIdleTimer: number | null = null;
  let buddyAnimMap: Record<string, string[]> = {};
  let buddyCache: BuddyData | null = null;
  let buddyAnimName = "idle";
  const buddyUserId = storedBuddyUserId();

  function moodFromChat(text?: string, role?: string): string {
    const t = (text || "").toLowerCase();
    if (!t) return "idle";
    if (role === "assistant") {
      if (/sorry|error|cannot|can't|unable/.test(t)) return "think";
      if (/!|great|done|success/.test(t)) return "happy";
      return "talk";
    }
    if (/haha|lol|lmao|😂|hehe/.test(t) || (t.match(/!/g) || []).length > 1) return "happy";
    if (/\b(why|how|what|explain|kyun|kaise|think)\b/.test(t) || t.indexOf("?") >= 0) return "think";
    if (/\b(bye|good night|sleep|so ja)\b/.test(t)) return "sleep";
    if (/\b(hi|hello|hey|namaste|hola)\b/.test(t)) return "wave";
    if (/\b(wow|amazing|yay)\b/.test(t)) return "excited";
    if (/\b(pet|love|cute)\b/.test(t)) return "pet";
    if (t.length > 120) return "think";
    return "blink";
  }

  function clearBuddyCssAnim(): void {
    if (!buddyFloat) return;
    CSS_ANIMS.forEach((n) => buddyFloat.classList.remove("anim-" + n));
    buddyFloat.classList.remove("petting");
  }

  function startBuddyIdle(ms?: number): void {
    stopBuddyIdle();
    const speed = ms || 700;
    buddyIdleTimer = window.setInterval(() => {
      if (!buddyFrames || !buddyFrames.length) return;
      buddyFrameIdx = (buddyFrameIdx + 1) % buddyFrames.length;
      const frame = buddyFrames[buddyFrameIdx];
      if (buddyAscii && frame !== undefined) buddyAscii.textContent = frame;
    }, speed);
  }

  function stopBuddyIdle(): void {
    if (buddyIdleTimer) window.clearInterval(buddyIdleTimer);
    buddyIdleTimer = null;
  }

  function setBuddyAnim(name?: string): void {
    buddyAnimName = name || "idle";
    clearBuddyCssAnim();
    if (buddyFloat) {
      if (CSS_ANIMS.indexOf(buddyAnimName) >= 0) buddyFloat.classList.add("anim-" + buddyAnimName);
      else buddyFloat.classList.add("anim-float");
    }
    const frames = (buddyAnimMap && buddyAnimMap[buddyAnimName]) || buddyFrames;
    if (frames && frames.length) {
      buddyFrames = frames;
      buddyFrameIdx = 0;
      if (buddyAscii && frames[0] !== undefined) buddyAscii.textContent = frames[0];
    }
    let speed = 700;
    if (buddyAnimName === "talk") speed = 280;
    if (buddyAnimName === "sleep") speed = 1400;
    if (buddyAnimName === "excited") speed = 320;
    startBuddyIdle(speed);
    // soft return to idle after a burst
    if (buddyAnimName !== "idle" && buddyAnimName !== "sleep") {
      window.setTimeout(() => {
        if (buddyOn && buddyAnimName !== "sleep") setBuddyAnim("idle");
      }, 4500);
    }
  }

  function setBuddyVisible(on: boolean): void {
    buddyOn = !!on;
    writeBuddyOn(buddyOn);
    const panel = document.getElementById("buddy-float");
    const ascii = document.getElementById("buddyAscii");
    const meta = document.getElementById("buddyMeta");
    if (panel) {
      if (buddyOn) {
        panel.classList.add("open");
        panel.style.display = "block";
        panel.setAttribute("aria-hidden", "false");
      } else {
        panel.classList.remove("open");
        panel.style.display = "none";
        panel.setAttribute("aria-hidden", "true");
      }
    }
    if (!buddyOn) {
      stopBuddyIdle();
      if (ascii) ascii.textContent = "";
      if (meta) meta.textContent = "";
      clearBuddyCssAnim();
    } else {
      loadBuddy().then(() => {
        setBuddyAnim("wave");
      });
    }
  }

  function localBuddyFallback(): void {
    const art = ["  /\\_/\\  ", " ( . . ) ", "  > ^ <  "].join("\n");
    if (buddyAscii) buddyAscii.textContent = art;
    if (buddyMeta) buddyMeta.textContent = "Buddy";
    buddyFrames = [art, art.replace(".", "-")];
    buddyAnimMap = { idle: buddyFrames, blink: buddyFrames, talk: buddyFrames };
  }

  function applyBuddyData(b: BuddyData | null | undefined): void {
    if (!b) return;
    if (buddyMeta) {
      buddyMeta.textContent = b.name ? b.name + " · " + (b.species_name || "") : "Buddy";
    }
    if (b.animations && typeof b.animations === "object") {
      buddyAnimMap = b.animations;
      buddyFrames = b.animations.idle || b.animations.blink || [];
    } else if (b.sprite_lines) {
      const art = b.sprite_lines.join("\n");
      buddyFrames = [art, art.replace(/o/, "-")];
      buddyAnimMap = { idle: buddyFrames, blink: buddyFrames };
    }
    if (buddyAscii && buddyFrames && buddyFrames[0] !== undefined) {
      buddyAscii.textContent = buddyFrames[0];
    }
  }

  async function loadBuddy(): Promise<void> {
    if (buddyCache && buddyCache.animations) {
      applyBuddyData(buddyCache);
      return;
    }
    try {
      const r = await fetch("/v1/buddy", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(window.__API_KEY ? { "X-API-Key": window.__API_KEY } : {}),
        },
        body: JSON.stringify({ command: "/buddy", user_id: buddyUserId }),
      });
      if (!r.ok) {
        localBuddyFallback();
        return;
      }
      const data = await r.json();
      const b = data.buddy || {};
      buddyCache = b;
      applyBuddyData(b);
      return;
    } catch (_) {
      localBuddyFallback();
    }
  }

  /** Call from chat pipeline so buddy reacts to AERIS conversation */
  window.sentinelBuddyReact = function (text, role) {
    if (!buddyOn) return;
    setBuddyAnim(moodFromChat(text, role || "user"));
  };

  /** Chat commands (/buddy on|off|pet) drive the float through meta_rag */
  window.sentinelBuddySetVisible = function (on) {
    setBuddyVisible(!!on);
  };
  window.sentinelBuddyPet = function () {
    if (!buddyOn) return;
    setBuddyAnim("pet");
  };

  if (buddyFloat) {
    const buddyIsTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
    if (buddyIsTouch) {
      // Touch: tap = pet; no accidental hide (use /buddy off to hide)
      buddyFloat.title = "Buddy — tap to pet · /buddy off to hide";
      buddyFloat.addEventListener("click", function () {
        if (buddyOn && window.sentinelBuddyPet) window.sentinelBuddyPet();
      });
    } else {
      buddyFloat.addEventListener("dblclick", function () {
        setBuddyVisible(false);
      });
    }
  }
  if (buddyOn) setBuddyVisible(true);
}
