/**
 * What a provider advertises about itself, mirroring ../python/capabilities.py.
 *
 * This is what lets the web app gain a model or a generation parameter without a code
 * change: the backend serves whatever Research published, and the UI renders a control
 * per `ParameterSpec`. There is no hardcoded parameter list on this side.
 */

export type ParameterType =
  "string" | "integer" | "number" | "boolean" | "enum";

export interface ParameterSpec {
  name: string;
  /** A type this client does not recognise should be treated as `string`. */
  type: ParameterType | string;
  label?: string | null;
  help?: string | null;
  minimum?: number | null;
  maximum?: number | null;
  max_length?: number | null;
  /** Allowed values, for `type: 'enum'`. */
  values?: string[];
  multiline?: boolean;
  /**
   * Which form the control belongs in when one provider serves several (character
   * swap: `create` options and per-`shot` edits). Absent means the only form.
   */
  group?: string | null;
  /** Suggested starting value; absent means take it from the shot file. */
  default?: unknown;
}

export interface ModelSpec {
  id: string;
  label?: string | null;
  default?: boolean;
  /** Specs that apply only to this model, merged over the provider list by name. */
  parameters?: ParameterSpec[];
}

export interface ProviderCapabilities {
  provider: string;
  models?: ModelSpec[];
  /** Accepted regardless of model. */
  parameters?: ParameterSpec[];
  note?: string | null;
  /** The job task this provider serves; absent means `generate`. */
  task?: string;
  updated_at?: string;
  contract_version?: string;
}

/** The providers that serve `task` (a provider without a task serves `generate`). */
export const providersForTask = (
  providers: ProviderCapabilities[],
  task: string,
): ProviderCapabilities[] =>
  providers.filter((provider) => (provider.task ?? "generate") === task);

/** The specs of one form `group`; `null` selects the specs that name no group. */
export const parametersInGroup = (
  specs: ParameterSpec[],
  group: string | null,
): ParameterSpec[] => specs.filter((spec) => (spec.group ?? null) === group);

/** `GET /api/capabilities`. `available` is false when no worker is advertising. */
export interface CapabilitiesResponse {
  available: boolean;
  providers: ProviderCapabilities[];
}

export const parameterLabel = (spec: ParameterSpec): string =>
  spec.label ??
  spec.name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

export const modelLabel = (model: ModelSpec): string => model.label ?? model.id;

export const defaultModel = (
  capabilities: ProviderCapabilities,
): ModelSpec | undefined =>
  capabilities.models?.find((model) => model.default) ??
  capabilities.models?.[0];

/**
 * Provider parameters with the chosen model's narrower specs merged in by name, matching
 * `ProviderCapabilities.parameters_for` on the Python side.
 *
 * The merge is field-level: a model that restates a parameter only to tighten a bound
 * keeps the shared label, help text and type. A spread is enough here because the
 * serialised form omits fields the author never set.
 */
export function parametersFor(
  capabilities: ProviderCapabilities,
  modelId: string | null | undefined,
): ParameterSpec[] {
  const base = capabilities.parameters ?? [];
  const model = capabilities.models?.find(
    (candidate) => candidate.id === modelId,
  );
  if (!model?.parameters?.length) return base;

  const narrowed = new Map(model.parameters.map((spec) => [spec.name, spec]));
  const merged = base.map((spec) => {
    const override = narrowed.get(spec.name);
    narrowed.delete(spec.name);
    return override ? { ...spec, ...override } : spec;
  });
  // Anything the model declares that the provider level does not.
  return [...merged, ...narrowed.values()];
}

/** Client-side mirror of `validate_override`, for inline feedback before submitting. */
export function validateOverride(
  spec: ParameterSpec,
  value: unknown,
): string | null {
  const label = parameterLabel(spec);

  if (spec.type === "boolean") {
    return typeof value === "boolean" ? null : `${label} must be true or false`;
  }

  if (spec.type === "integer" || spec.type === "number") {
    if (typeof value !== "number" || !Number.isFinite(value))
      return `${label} must be a number`;
    if (spec.type === "integer" && !Number.isInteger(value))
      return `${label} must be a whole number`;
    if (spec.minimum != null && value < spec.minimum)
      return `${label} must be at least ${spec.minimum}`;
    if (spec.maximum != null && value > spec.maximum)
      return `${label} must be at most ${spec.maximum}`;
    return null;
  }

  if (spec.type === "enum") {
    return spec.values?.includes(String(value))
      ? null
      : `${label} must be one of the listed values`;
  }

  if (typeof value !== "string") return `${label} must be text`;
  if (spec.max_length != null && value.length > spec.max_length)
    return `${label} is longer than the ${spec.max_length} character limit`;
  return null;
}
