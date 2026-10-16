/**
 * SENTINEL product model nicknames + effort.
 * One backend checkpoint for now; UI maps nickname → tokens / temperature.
 */

export type ModelId = "blink" | "fable" | "mythos" | "oracle";
export type EffortId = "low" | "medium" | "recommend" | "high" | "extra" | "max";

export interface ModelDef {
  id: ModelId;
  name: string;
  tagline: string;
  defaultEffort: EffortId;
  temperature: number;
}

export interface EffortDef {
  id: EffortId;
  label: string;
  maxTokens: number;
  blurb: string;
}

/** Own nicknames — not third-party model brands. */
export const MODELS: ModelDef[] = [
  {
    id: "blink",
    name: "Blink",
    tagline: "Tez jawab · short chat",
    defaultEffort: "low",
    temperature: 0.5,
  },
  {
    id: "fable",
    name: "Fable",
    tagline: "Balanced · seedha · product default",
    defaultEffort: "recommend",
    temperature: 0.4,
  },
  {
    id: "mythos",
    name: "Mythos",
    tagline: "Zyada soch · lambe jawab",
    defaultEffort: "high",
    temperature: 0.35,
  },
  {
    id: "oracle",
    name: "Oracle",
    tagline: "Max effort · mushkil kaam",
    defaultEffort: "max",
    temperature: 0.3,
  },
];

export const EFFORTS: EffortDef[] = [
  { id: "low", label: "Low", maxTokens: 48, blurb: "Short · quick" },
  { id: "medium", label: "Medium", maxTokens: 80, blurb: "Normal length" },
  { id: "recommend", label: "Recommend", maxTokens: 120, blurb: "Suggested default" },
  { id: "high", label: "High", maxTokens: 192, blurb: "More depth" },
  { id: "extra", label: "Extra", maxTokens: 320, blurb: "Long answers" },
  { id: "max", label: "Max", maxTokens: 512, blurb: "Full budget (slow on CPU)" },
];

export const DEFAULT_MODEL: ModelId = "fable";
export const DEFAULT_EFFORT: EffortId = "recommend";

export function modelById(id: string | undefined | null): ModelDef {
  return MODELS.find((m) => m.id === id) || MODELS.find((m) => m.id === DEFAULT_MODEL)!;
}

export function effortById(id: string | undefined | null): EffortDef {
  return EFFORTS.find((e) => e.id === id) || EFFORTS.find((e) => e.id === DEFAULT_EFFORT)!;
}

export function resolveGenParams(modelId: ModelId, effortId: EffortId): {
  maxNewTokens: number;
  temperature: number;
  modelLabel: string;
  effortLabel: string;
} {
  const m = modelById(modelId);
  const e = effortById(effortId);
  return {
    maxNewTokens: e.maxTokens,
    temperature: m.temperature,
    modelLabel: m.name,
    effortLabel: e.label,
  };
}
