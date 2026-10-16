/**
 * Buddy float controller — imperative port of the legacy inline script.
 * Same DOM ids (#buddy-float, #buddyAscii, #buddyMeta), same window hooks
 * (sentinelBuddyReact / sentinelBuddySetVisible / sentinelBuddyPet / sentinelBuddyStep).
 *
 * Pokémon-style egg stage: while the server reports an unhatched egg the float
 * shows egg frames + incubation progress instead of the companion. Every user
 * chat message is one step (App calls window.sentinelBuddyStep()), passive time
 * also counts; on hatch the float plays the crack sequence and reveals the buddy.
 */
import { buddyUserId as storedBuddyUserId, buddyOnChosen, readBuddyOn, writeBuddyOn } from "../../buddy";

const CSS_ANIMS = ["float", "talk", "think", "happy", "sleep", "excited", "wave", "pet"];

interface BuddyData {
  name?: string;
  species_name?: string;
  shiny?: boolean;
  animations?: Record<string, string[]>;
  sprite_lines?: string[];
}

interface EggState {
  messages?: number;
  progress?: number;
  required?: number;
  hatched?: boolean;
  period_s?: number;
  next_step_in_s?: number;
  frames?: string[];
  crack_frames?: string[];
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

  // ---- egg incubation state ------------------------------------------------
  let egg: EggState | null = null; // unhatched snapshot; null = no egg / hatched
  let eggBaseAt = 0; // Date.now() when the snapshot arrived
  let eggTickTimer: number | null = null;
  let eggIdleSpeed = 800;
  let hatching = false;

  /** Companion identity: signed-in account, else per-browser device id. */
  function buddyId(): string {
    return storedBuddyUserId();
  }

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
    if (egg && !egg.hatched) return; // egg keeps wobbling; no mood anims
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

  // ---- egg helpers -----------------------------------------------------------

  function eggProgressNow(): number {
    if (!egg) return 0;
    const req = egg.required || 1;
    if (egg.hatched) return req;
    const period = Math.max(1, (egg.period_s || 300) * 1000);
    const lead = Math.max(0, period - (egg.next_step_in_s || 0) * 1000);
    const since = Math.max(0, Date.now() - eggBaseAt);
    const timePart = Math.floor((lead + since) / period);
    return Math.min((egg.progress || 0) + timePart, req);
  }

  function eggNextInS(): number {
    if (!egg) return 0;
    const period = Math.max(1, (egg.period_s || 300) * 1000);
    const lead = Math.max(0, period - (egg.next_step_in_s || 0) * 1000);
    const since = Math.max(0, Date.now() - eggBaseAt);
    return Math.ceil((period - ((lead + since) % period)) / 1000);
  }

  function fmtEta(s: number): string {
    if (s <= 1) return "now";
    if (s >= 90) return Math.round(s / 60) + "m";
    return s + "s";
  }

  function eggMetaText(): string {
    if (!egg) return "Egg";
    const p = eggProgressNow();
    const r = egg.required || 1;
    if (p >= r) return "Egg · ready! hatching…";
    return "Egg · " + p + "/" + r + " steps · next ~" + fmtEta(eggNextInS());
  }

  /** Wobble speed rises as the egg gets closer to hatching. */
  function eggSpeedFor(p: number, r: number): number {
    if (p >= r) return 240;
    const frac = Math.min(1, Math.max(0, p / Math.max(1, r)));
    return Math.round(950 - 600 * frac);
  }

  function stopEggTicker(): void {
    if (eggTickTimer) window.clearInterval(eggTickTimer);
    eggTickTimer = null;
  }

  function startEggTicker(): void {
    stopEggTicker();
    eggTickTimer = window.setInterval(() => {
      if (!egg || egg.hatched || hatching) return;
      const p = eggProgressNow();
      const r = egg.required || 1;
      if (p >= r) {
        hatchEgg();
        return;
      }
      if (buddyMeta) buddyMeta.textContent = eggMetaText();
      updateProgressBar(p, r);
      const speed = eggSpeedFor(p, r);
      if (speed !== eggIdleSpeed) {
        eggIdleSpeed = speed;
        startBuddyIdle(speed);
      }
    }, 3000);
  }

  function updateProgressBar(p: number, r: number): void {
    const wrap = document.getElementById("buddyProgress");
    const bar = document.getElementById("buddyProgressBar");
    if (!wrap || !bar) return;
    wrap.hidden = false;
    const pct = Math.min(100, Math.round((100 * p) / Math.max(1, r)));
    bar.style.width = pct + "%";
  }

  function enterEggMode(data: EggState, soft = false): void {
    if (!data || data.hatched) return;
    egg = data;
    eggBaseAt = Date.now();
    if (hatching) return; // don't yank frames mid-hatch
    const frames = data.frames && data.frames.length ? data.frames : buddyFrames;
    if (frames && frames.length) {
      buddyFrames = frames;
      if (!soft) {
        buddyFrameIdx = 0;
        if (buddyAscii && frames[0] !== undefined) buddyAscii.textContent = frames[0];
      } else if (buddyAscii) {
        const f = frames[buddyFrameIdx];
        if (f !== undefined) buddyAscii.textContent = f;
      }
    }
    const p = eggProgressNow();
    const r = data.required || 1;
    
    if (buddyMeta) buddyMeta.textContent = p >= r ? "Egg · ready!" : eggMetaText();
    updateProgressBar(p, r);
    eggIdleSpeed = eggSpeedFor(p, r);
    startBuddyIdle(eggIdleSpeed);
    startEggTicker();
  }

  /** Crack sequence → revealed companion (freshBuddy skips the re-fetch). */
  function hatchEgg(freshBuddy?: BuddyData | null): void {
    if (hatching) return;
    hatching = true;
    stopEggTicker();
    stopBuddyIdle();
    const cracks = (egg && egg.crack_frames) || [];
    let i = 0;
    const finish = () => {
      egg = null;
      
      const wrap = document.getElementById("buddyProgress");
      if (wrap) wrap.hidden = true;
      if (freshBuddy) {
        hatching = false;
        buddyCache = freshBuddy;
        applyBuddyData(freshBuddy);
        setBuddyAnim("wave");
        return;
      }
      hatching = false;
      void loadBuddy().then(() => {
        if (buddyOn) setBuddyAnim("wave");
      });
    };
    const step = () => {
      if (i < cracks.length) {
        const f = cracks[i];
        if (buddyAscii && f !== undefined) buddyAscii.textContent = f;
        i += 1;
        window.setTimeout(step, 430);
        return;
      }
      finish();
    };
    step();
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
      stopEggTicker();
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
      if (b.shiny) buddyMeta.textContent += " ✨";
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
    if (buddyCache && buddyCache.animations && !egg) {
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
        body: JSON.stringify({ command: "/buddy", user_id: buddyId() }),
      });
      if (!r.ok) {
        localBuddyFallback();
        return;
      }
      const data = await r.json();
      const eggData = data.egg as EggState | undefined;
      if (eggData && !eggData.hatched) {
        enterEggMode(eggData);
        return;
      }
      egg = null;
      stopEggTicker();
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
    if (egg && !egg.hatched && !hatching) {
      // Petting the egg = excited wobble burst
      startBuddyIdle(180);
      window.setTimeout(() => {
        if (egg && !egg.hatched && !hatching) startBuddyIdle(eggIdleSpeed || 800);
      }, 900);
      return;
    }
    setBuddyAnim("pet");
  };

  /**
   * One incubation step — App calls this after each real user message.
   * Response flips to `hatched` near the goal, which triggers the reveal.
   */
  window.sentinelBuddyStep = function () {
    try {
      void fetch("/v1/buddy/step", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(window.__API_KEY ? { "X-API-Key": window.__API_KEY } : {}),
        },
        body: JSON.stringify({ user_id: buddyId() }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data: { hatched?: boolean; egg?: EggState; buddy?: BuddyData } | null) => {
          if (!data) return;
          if (data.egg && !data.egg.hatched) {
            if (buddyOn) enterEggMode(data.egg, true);
            return;
          }
          if (!data.hatched) return;
          const b = data.buddy || null;
          if (!buddyOn) {
            if (b) buddyCache = b; // hidden float: reveal happens on next open
            return;
          }
          if (egg && !hatching) {
            hatchEgg(b);
          } else if (b) {
            buddyCache = b;
            applyBuddyData(b);
            setBuddyAnim("wave");
          }
        })
        .catch(() => {});
    } catch (_) {
      /* never block chat on the pet */
    }
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
  if (buddyOn) {
    setBuddyVisible(true);
  } else if (!buddyOnChosen()) {
    // First visit: the egg is the discovery moment — pop the float just once.
    void loadBuddy().then(() => {
      if (egg && !egg.hatched) setBuddyVisible(true);
    });
  }
}
