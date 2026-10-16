/**
 * SENTINEL product model lineup + effort (Claude-style clarity, own branding).
 * Client maps these to max_new_tokens / temperature — one backend checkpoint for now.
 */

export type ModelId = "haiku" | "fable" | "sonnet" | "opus";
export type EffortId = "low" | "medium" | "recommend" | "high" | "extra" | "max";

export interface ModelDef {
  id: ModelId;
  name: string;
  tagline: string;
  /** Default effort when this model is selected */
  defaultEffort: EffortId;
  /** Temperature bias */
  temperature: number;
}

export interface EffortDef {
  id: EffortId;
  label: string;
  /** Max new tokens sent to the API */
  maxTokens: number;
  blurb: string;
}

export const MODELS: ModelDef[] = [
  {
    id: "haiku",
    name: "Haiku",
    tagline: "Fast replies · everyday chat",
    defaultEffort: "low",
    temperature: 0.5,
  },
  {
    id: "fable",
    name: "Fable",
    tagline: "Balanced · story-clear · product default",
    defaultEffort: "recommend",
    temperature: 0.4,
  },
  {
    id: "sonnet",
    name: "Sonnet",
    tagline: "Stronger reasoning · longer answers",
    defaultEffort: "high",
    temperature: 0.35,
  },
  {
    id: "opus",
    name: "Opus",
    tagline: "Maximum effort · complex tasks",
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

/** Resolve API generation params from product UI selection. */
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
