/**
 * Original Buddy float — ASCII/sprite only (no toolbar / settings clutter).
 * Hatching + moods on this float, never inside the chat list.
 */
export function BuddyFloat() {
  return (
    <aside
      id="buddy-float"
      aria-label="Buddy companion — linked to you and AERIS-64"
      aria-hidden="true"
      title="Buddy — double-click to hide · /buddy off"
    >
      <div className="buddy-brand" aria-hidden="true">
        AERIS-64 · Buddy
      </div>
      <div className="buddy-meta" id="buddyMeta"></div>
      <div className="buddy-ascii" id="buddyAscii" role="img" aria-live="polite" aria-label="Buddy sprite"></div>
      <div className="buddy-progress" id="buddyProgress" hidden>
        <div className="buddy-progress-bar" id="buddyProgressBar" />
      </div>
    </aside>
  );
}
