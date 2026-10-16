/**
 * Original Buddy float — ASCII/sprite only (no extra smiley orb).
 * Hatching + moods on this float, never inside the chat list.
 * Behavior: lib/buddy-float.ts
 */
export function BuddyFloat() {
  return (
    <aside
      id="buddy-float"
      aria-label="Buddy companion — linked to you and AERIS-64"
      aria-hidden="true"
      title="Buddy — double-click to hide · /buddy off"
    >
      <div className="buddy-brand" aria-hidden="true">AERIS-64 · Buddy</div>
      <div className="buddy-meta" id="buddyMeta"></div>
      <div className="buddy-ascii" id="buddyAscii"></div>
      <div className="buddy-progress" id="buddyProgress" hidden>
        <div className="buddy-progress-bar" id="buddyProgressBar" />
      </div>
    </aside>
  );
}
