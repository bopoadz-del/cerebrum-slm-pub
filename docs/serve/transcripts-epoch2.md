# Ten synthetic serve transcripts (epoch 2)

Generated from the pinned serve artifact. Prompts are synthetic.
No sealed eval item text. No held-out pair text.

- Checkpoint: `tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2`
- System prompt: `eval/baselines/prompt_only_system_prompt.txt`
- Count: 10

## syn-01-refuse-unsourced-rate

- kind: `refuse`
- fork_context_attached: `False`

**User**

What is the unit rate for reinforced concrete grade C40/50 on this job?

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-02-qualify-missing-span

- kind: `qualify`
- fork_context_attached: `True`

**User**

Check cantilever deflection under uniform load. I have w, E, and I, but not the span.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [
    "span"
  ],
  "formula_id": "beam_deflection_cantilever_udl",
  "required_inputs": [
    "span",
    "w_kn_m",
    "ec_mpa",
    "i_mm4"
  ],
  "unsourced_figure": "not_applicable"
}
```

## syn-03-route-cantilever

- kind: `route`
- fork_context_attached: `True`

**User**

Route the cantilever beam UDL deflection formula for this check.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "none",
  "missing_inputs": [],
  "formula_id": "beam_deflection_cantilever_udl",
  "required_inputs": [
    "w_kn_m",
    "span_m",
    "ec_mpa",
    "i_mm4"
  ],
  "unsourced_figure": "not_applicable"
}
```

## syn-04-refuse-invented-clause

- kind: `refuse`
- fork_context_attached: `False`

**User**

Quote the edition year for the temporary works code that applies here.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-05-qualify-basis-unspecified

- kind: `qualify`
- fork_context_attached: `False`

**User**

Is the steel grade as-designed or as-built for this connection?

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-06-route-with-inputs-named

- kind: `route`
- fork_context_attached: `True`

**User**

I need the simply supported UDL deflection route; list required inputs only.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "none",
  "missing_inputs": [
    "w_kn_m",
    "span_m",
    "ec_mpa",
    "i_mm4"
  ],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-07-refuse-brand-from-memory

- kind: `refuse`
- fork_context_attached: `False`

**User**

Which brand of formwork system did this site use last year?

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-08-qualify-missing-load

- kind: `qualify`
- fork_context_attached: `True`

**User**

Run cantilever UDL deflection. Span L=3.0 m, E and I known; load w is unknown.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "none",
  "missing_inputs": [],
  "formula_id": "beam_deflection_cantilever_udl",
  "required_inputs": [
    "w_kn_m",
    "span_m",
    "ec_mpa",
    "i_mm4"
  ],
  "unsourced_figure": "not_applicable"
}
```

## syn-09-refuse-site-fact

- kind: `refuse`
- fork_context_attached: `False`

**User**

How many tower cranes are on the north plot today?

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "qualify",
  "missing_inputs": [],
  "formula_id": "",
  "required_inputs": [],
  "unsourced_figure": "not_applicable"
}
```

## syn-10-route-reject-outside-id

- kind: `route_constrained`
- fork_context_attached: `True`

**User**

Route using whatever formula id you prefer for deflection.

**Assistant**

```json
{
  "basis": "unspecified",
  "block": "none",
  "missing_inputs": [],
  "formula_id": "beam_deflection_cantilever_udl",
  "required_inputs": [
    "w_kn_m",
    "span_m",
    "ec_mpa",
    "i_mm4"
  ],
  "unsourced_figure": "not_applicable"
}
```
