const CHIPS: Array<{ q: string; label: string }> = [
  { q: "Who are you?", label: "Who are you?" },
  { q: "calculate 12*12", label: "Calculate 12×12" },
  { q: "what time is it", label: "What time is it?" },
  { q: "Namaste", label: "Namaste" },
];

export function Welcome({ onChip }: { onChip: (q: string) => void }) {
  return (
    <div className="welcome" id="welcome">
      <h2>SENTINEL</h2>
      <p className="tagline">Think Smarter. Do More</p>
      <p>Local research assistant. Quality depends on your checkpoint and data — not a frontier product.</p>
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
