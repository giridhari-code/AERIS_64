/**
 * Buddy companion float — static shell here, behavior wired by the imperative
 * controller in lib/buddy-float.ts (same DOM ids/classes as the legacy page).
 * No React-managed props on the dynamic nodes, so imperative updates stick.
 */
export function BuddyFloat() {
  return (
    <aside
      id="buddy-float"
      aria-label="Buddy companion"
      aria-hidden="true"
      title="Double-click to hide"
    >
      <div className="buddy-meta" id="buddyMeta"></div>
      <div className="buddy-ascii" id="buddyAscii"></div>
    </aside>
  );
}
