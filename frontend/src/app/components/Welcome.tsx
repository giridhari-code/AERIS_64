const CHIPS: Array<{ q: string; label: string }> = [
  { q: "Who are you?", label: "Who are you?" },
  { q: "Namaste", label: "Namaste" },
  { q: "what time is it", label: "Time" },
];

export function Welcome({
  onChip,
  userName,
}: {
  onChip: (q: string) => void;
  userName?: string | null;
}) {
  const hi = userName ? `Hi ${userName.split(" ")[0]}` : "Hi";
  return (
    <div className="welcome" id="welcome">
      <h2>{hi}.</h2>
      <p className="tagline">Seedha jawab. No chapluchi.</p>
      <div className="chips">
        {CHIPS.map((c) => (
          <button key={c.q} type="button" className="chip" onClick={() => onChip(c.q)}>
            {c.label}
          </button>
        ))}
      </div>
    </div>
  );
}
