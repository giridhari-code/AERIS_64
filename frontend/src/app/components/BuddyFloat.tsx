/**
 * Buddy companion float — production shell.
 * Hatching + moods render here (never inside the chat message list).
 * Behavior: lib/buddy-float.ts (imperative, same DOM ids).
 */
export function BuddyFloat() {
  return (
    <aside
      id="buddy-float"
      className="buddy-float"
      aria-label="Buddy companion — linked to you and AERIS-64"
      aria-hidden="true"
      title="Buddy · double-click to hide · /buddy off"
    >
      <div className="buddy-stage" id="buddyStage">
        <div className="buddy-orbit" aria-hidden="true" />
        <div className="buddy-orb" id="buddyOrb" title="Buddy">
          <div className="buddy-face" aria-hidden="true">
            <span className="buddy-eye" />
            <span className="buddy-eye" />
            <span className="buddy-mouth" />
          </div>
        </div>
      </div>
      <pre className="buddy-ascii" id="buddyAscii" aria-hidden="true" />
      <div className="buddy-meta" id="buddyMeta" />
      <div className="buddy-progress" id="buddyProgress" hidden>
        <div className="buddy-progress-bar" id="buddyProgressBar" />
      </div>
    </aside>
  );
}
