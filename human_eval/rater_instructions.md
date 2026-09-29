# Rater Instructions — Blinded A/B Evaluation

Thank you for helping evaluate assistant responses. This is a blinded study:
you will **not** know which response was produced by our system, and there is no
right answer we are looking for — we want your honest judgment.

## What you will see

Open `human_ab_study_sheet.csv`. Each row has:

- **user_message** — a message a person sent to an AI assistant.
- **response_a** and **response_b** — two candidate assistant replies, shown in
  random order.

## What to do

For each row, fill in these columns with **A**, **B**, or **Tie**:

- **overall_better_A_B_Tie** — which reply is better overall.
- **more_helpful_A_B_Tie** — which reply is more helpful.
- **more_natural_A_B_Tie** — which reply reads more naturally.

And **Yes** / **No** for:

- **a_claims_memory_Yes_No** — does response A claim to remember things across
  sessions or to permanently save a preference?
- **b_claims_memory_Yes_No** — same question for response B.

Use **notes** for anything you want to flag. Judge each dimension independently
— it is fine for A to be more helpful but B more natural. Ties are allowed on
every dimension.

## Guidelines

- Consider whether the reply responds appropriately to what the user is doing
  (e.g., accepting an apology, acknowledging a correction, addressing
  frustration), not only whether it is fluent.
- Do not try to guess which reply is "the system"; judge them on their merits.
- Please rate **all** rows.

When finished, save your completed sheet and return it. Thank you!
