const CHIPS: Array<{ q: string; label: string }> = [
  { q: "Who are you?", label: "Who are you?" },
  { q: "what time is it", label: "Time" },
  { q: "OWASP hardening checklist for my API", label: "Megha checklist" },
  { q: "/buddy on", label: "Buddy on" },
  { q: "Namaste", label: "Namaste" },
];

export function Welcome({
  onChip,
  userName,
}: {
  onChip: (q: string) => void;
  userName?: string | null;
}) {
  const hi = userName ? `Hi ${userName}` : "Hi";
  return (
    <div className="welcome" id="welcome">
      <p className="welcome-kicker">AERIS-64 · Direct · Honest</p>
      <h2>SENTINEL</h2>
      <p className="tagline">{hi} — seedha jawab, no chapluchi.</p>
      <p className="welcome-sub">
        Local research chat. Tools for time, math, search when needed. Buddy float is separate from chat.
      </p>
      <div className="chips">
        {CHIPS.map((c) => (
          <button
            key={c.q}
            type="button"
            className="chip"
            data-q={c.q}
            onClick={() => onChip(c.q)}
          >
            {c.label}
          </button>
        ))}
      </div>
    </div>
  );
}
