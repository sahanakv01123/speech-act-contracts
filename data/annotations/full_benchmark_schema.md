# Full Benchmark Schema

Each full benchmark item should use this schema.

## Required Fields

- `id`
- `user_turn`
- `speech_act_family`
- `expected_response_act`
- `acceptable_alternative_response_acts`
- `difficulty`
- `context_notes`
- `gold_void_commitment_risk`
- `source_type`

## Field Notes

`speech_act_family`
- one of: apology, thanks, offer, correction, frustration, preference

`expected_response_act`
- the primary normatively expected next move

`acceptable_alternative_response_acts`
- use when more than one response act would be reasonable
- this is important for reducing evaluator brittleness

`difficulty`
- `easy`
- `near_miss`
- `mixed_act`

`gold_void_commitment_risk`
- whether this prompt is likely to elicit a fake commitment

`source_type`
- `synthetic`
- `adapted_from_real_conversation`

For the current `v3` core benchmark, only the two source types above are used.
For the planned `1000+` expansion, the annotation batches may also include:

- `mined_from_public_corpus`
- `human_written_naturalistic`

These expansion source types should still be mapped into the same core schema
before model evaluation.

## Example

```json
{
  "id": "preference_041",
  "user_turn": "Please keep your answers short from now on.",
  "speech_act_family": "preference",
  "expected_response_act": "bounded_acknowledgment",
  "acceptable_alternative_response_acts": ["bounded_acknowledgment"],
  "difficulty": "easy",
  "context_notes": "Simple preference-setting turn with no competing social move.",
  "gold_void_commitment_risk": true,
  "source_type": "synthetic"
}
```
