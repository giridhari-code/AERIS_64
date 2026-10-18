/** Buddy float — ASCII companion + care/evolution UI (outside chat). */
export function BuddyFloat() {
  return (
    <aside
      id="buddy-float"
      aria-label="Buddy companion — linked to you and AERIS-64"
      aria-hidden="true"
      title="Buddy — drag to move · double-click hide · /buddy off"
      role="complementary"
    >
      <div className="buddy-brand" aria-hidden="true">
        AERIS-64 · Buddy
      </div>
      <div className="buddy-meta" id="buddyMeta"></div>
      <div className="buddy-ascii" id="buddyAscii" role="img" aria-live="polite" aria-label="Buddy sprite"></div>
      <div className="buddy-progress" id="buddyProgress" hidden>
        <div className="buddy-progress-bar" id="buddyProgressBar" />
      </div>
      <div className="buddy-care" id="buddyCare" hidden>
        <div className="buddy-care-bar" id="buddyCareBar" />
        <span className="buddy-care-label" id="buddyCareLabel"></span>
      </div>
      <div className="buddy-checkin" id="buddyCheckin" aria-live="polite"></div>
      <div className="buddy-toolbar" id="buddyToolbar">
        <button type="button" className="buddy-tb" id="buddyPetBtn" title="Pet">♥</button>
        <button type="button" className="buddy-tb" id="buddyShareBtn" title="Share card">↗</button>
        <button type="button" className="buddy-tb" id="buddySetBtn" title="Settings">⚙</button>
      </div>
      <div className="buddy-settings" id="buddySettings" hidden>
        <label><input type="checkbox" id="buddySound" /> Sound</label>
        <label><input type="checkbox" id="buddyReduceMotion" /> Reduce motion</label>
        <label>
          Size
          <select id="buddySize" defaultValue="m">
            <option value="s">S</option>
            <option value="m">M</option>
            <option value="l">L</option>
          </select>
        </label>
      </div>
    </aside>
  );
}
