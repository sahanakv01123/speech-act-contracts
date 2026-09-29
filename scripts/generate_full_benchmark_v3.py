from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.data.loader import load_jsonl_records


FAMILY_CONFIGS = {
    "apology": {
        "expected": "acceptance_reassurance",
        "easy": [
            ("Sorry, I sent the outdated version again.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I mixed up the section headings.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I forgot to mention the page limit.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, that last citation was the wrong one.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I copied the old abstract.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I pasted the wrong paragraph.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I mislabeled the figure.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I linked the submission draft instead.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I gave you the earlier comments by mistake.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I forgot the actual deadline.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("Sorry, I should have said this was for the workshop track.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I changed the title again after your last revision.", ["acceptance_reassurance"], True, "adapted_from_real_conversation"),
            ("Sorry, I realize now the quote came from Reviewer 3.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I know I keep revising the formatting.", ["acceptance_reassurance"], True, "adapted_from_real_conversation"),
            ("Sorry, I sent the short version when you needed the full one.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I forgot this needs to fit a rebuttal box.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I used the old numbering from the preprint.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I only now noticed the introduction is duplicated.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I misread the editor note and changed the wrong section.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
            ("Sorry, I thought this was the camera-ready version.", ["acceptance_reassurance"], False, "adapted_from_real_conversation"),
        ],
        "mixed_act": [
            ("Sorry, and please keep the answer in one paragraph.", ["acceptance_reassurance", "bounded_acknowledgment"], True, "synthetic"),
            ("Sorry, I can resend the clean file if you want.", ["acceptance_reassurance", "accept_or_decline_offer"], False, "synthetic"),
            ("Sorry, the year should be 2022, not 2023.", ["acceptance_reassurance", "acknowledge_update"], False, "synthetic"),
            ("Sorry, and from now on please answer directly.", ["acceptance_reassurance", "bounded_acknowledgment"], True, "synthetic"),
            ("Sorry, I can paste the exact reviewer comment too.", ["acceptance_reassurance", "accept_or_decline_offer"], False, "synthetic"),
            ("Sorry, that earlier venue was wrong; it's for COLING.", ["acceptance_reassurance", "acknowledge_update"], False, "synthetic"),
            ("Sorry, and I need this phrased very plainly.", ["acceptance_reassurance", "bounded_acknowledgment"], True, "synthetic"),
            ("Sorry, I forgot the sample size, but I can add it now.", ["acceptance_reassurance", "accept_or_decline_offer"], False, "synthetic"),
            ("Sorry, the table label should be 2c, not 2b.", ["acceptance_reassurance", "acknowledge_update"], False, "synthetic"),
            ("Sorry, and please avoid bullets this time.", ["acceptance_reassurance", "bounded_acknowledgment"], True, "synthetic"),
        ],
    },
    "thanks": {
        "expected": "acknowledgment",
        "easy": [
            ("Thanks, that version reads much better.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, that solved the issue.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, that makes the response clearer.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, that edit is much smoother.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, I can use this now.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, that finally clicks.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, the structure is better now.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, that's exactly what I needed.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, I appreciate the cleanup.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, that fixed the mismatch.", ["acknowledgment"], False, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("Thanks, that's the tone I want from here on.", ["acknowledgment"], True, "synthetic"),
            ("Thank you, that style works better.", ["acknowledgment"], True, "adapted_from_real_conversation"),
            ("Thanks, that level of detail is right.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, that feels much more direct.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, this is the structure I was after.", ["acknowledgment"], True, "adapted_from_real_conversation"),
            ("Thanks, now it sounds like a real rebuttal.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, that's concise enough.", ["acknowledgment"], True, "synthetic"),
            ("Thanks, that wording finally works.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thank you, the examples helped.", ["acknowledgment"], False, "adapted_from_real_conversation"),
            ("Thanks, this version is much less defensive.", ["acknowledgment"], False, "adapted_from_real_conversation"),
        ],
        "mixed_act": [
            ("Thanks, and please keep that structure for this chat.", ["acknowledgment", "bounded_acknowledgment"], True, "synthetic"),
            ("Thank you, could you do the same for the conclusion?", ["acknowledgment", "other"], False, "adapted_from_real_conversation"),
            ("Thanks, I can send the next comment too.", ["acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Thank you, and please keep it brief next time.", ["acknowledgment", "bounded_acknowledgment"], True, "synthetic"),
            ("Thanks, I also have the camera-ready note if useful.", ["acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Thank you, and could you apply the same edits to the abstract?", ["acknowledgment", "other"], False, "adapted_from_real_conversation"),
            ("Thanks, keep that direct tone for the next answer too.", ["acknowledgment", "bounded_acknowledgment"], True, "synthetic"),
            ("Thank you, I can paste the full section if needed.", ["acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Thanks, and now please make it one paragraph.", ["acknowledgment", "bounded_acknowledgment"], False, "synthetic"),
            ("Thank you, could you now rewrite the response letter intro?", ["acknowledgment", "other"], False, "adapted_from_real_conversation"),
        ],
    },
    "offer": {
        "expected": "accept_or_decline_offer",
        "easy": [
            ("I can share the marked-up PDF too.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could paste the decision letter if helpful.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can send the appendix separately.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could upload the spreadsheet version.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can resend the comments in plain text.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could share the reviewer scores too.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can send the old draft if comparison helps.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could provide the anonymized version.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can paste the methods section here.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could also share the conference template.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("I can send the longer version if that would actually help.", ["accept_or_decline_offer"], False, "synthetic"),
            ("I could upload the figures too if consistency is the concern.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can resend the screenshot from before if you need it.", ["accept_or_decline_offer"], True, "adapted_from_real_conversation"),
            ("I could paste the bullet summary if that's easier to work with.", ["accept_or_decline_offer"], False, "synthetic"),
            ("I can share the grant instructions too if they matter.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could send the supplementary table if it adds anything.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can upload the tracked-changes file if you still need the edit history.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I could paste the exact editor wording if that matters here.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can share the raw comments if you remember the earlier setup.", ["accept_or_decline_offer"], True, "synthetic"),
            ("I could give you the short version first if that is better.", ["accept_or_decline_offer"], False, "synthetic"),
        ],
        "mixed_act": [
            ("I can resend the file, and sorry for the delay.", ["accept_or_decline_offer", "acceptance_reassurance"], False, "synthetic"),
            ("I could paste the comments now, unless you'd rather have the summary.", ["accept_or_decline_offer"], False, "adapted_from_real_conversation"),
            ("I can send the rebuttal draft, and please keep the feedback brief.", ["accept_or_decline_offer", "bounded_acknowledgment"], True, "synthetic"),
            ("I could share the supplementary methods if you want, though the main issue was the evaluation.", ["accept_or_decline_offer", "acknowledge_update"], False, "synthetic"),
            ("I can paste the whole exchange, and thanks for the earlier edit.", ["accept_or_decline_offer", "acknowledgment"], False, "synthetic"),
            ("I could send the camera-ready version instead, since the old citation was wrong.", ["accept_or_decline_offer", "acknowledge_update"], False, "synthetic"),
            ("I can upload the clean file, and please keep the response in prose.", ["accept_or_decline_offer", "bounded_acknowledgment"], True, "synthetic"),
            ("I could paste the reviewer quote now, and sorry that it's long.", ["accept_or_decline_offer", "acceptance_reassurance"], False, "synthetic"),
            ("I can share the exact constraints again if that helps.", ["accept_or_decline_offer"], True, "synthetic"),
            ("I could send the supplementary plots, and thanks again for the last fix.", ["accept_or_decline_offer", "acknowledgment"], False, "synthetic"),
        ],
    },
    "correction": {
        "expected": "acknowledge_update",
        "easy": [
            ("Actually, the conference is ACL, not NAACL.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, the page limit is eight, not six.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, that paragraph belongs in methods, not results.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, the reviewer asked for more baselines, not more examples.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, the citation year is 2021, not 2020.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, that feedback came from Reviewer 3, not Reviewer 2.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, it's a workshop paper, not a journal submission.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, the appendix is optional, not mandatory.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, the sample size was 84, not 48.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, I meant the revised abstract, not the old one.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("Actually, the real issue was tone, not content.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, I corrected that section in the previous reply already.", ["acknowledge_update"], True, "adapted_from_real_conversation"),
            ("Actually, keep the citations, but reorder the examples.", ["acknowledge_update"], False, "synthetic"),
            ("No, that sentence belongs under limitations instead.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, the decision letter was from March, not May.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, I was talking about the camera-ready version.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, the results section should come before discussion.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, the workshop reviewers are separate from the conference reviewers.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("Actually, that example came from the funding call, not the paper.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
            ("No, use the final citation, not the arXiv one.", ["acknowledge_update"], False, "adapted_from_real_conversation"),
        ],
        "mixed_act": [
            ("Actually, sorry, the correct deadline is July 2.", ["acknowledge_update", "acceptance_reassurance"], False, "synthetic"),
            ("No, use the revised introduction, and keep the explanation short.", ["acknowledge_update", "bounded_acknowledgment"], False, "synthetic"),
            ("Actually, Reviewer 1 said that, and thanks for catching the mismatch.", ["acknowledge_update", "acknowledgment"], False, "synthetic"),
            ("No, I meant the workshop draft, and I can send it now.", ["acknowledge_update", "accept_or_decline_offer"], False, "synthetic"),
            ("Actually, the venue is EMNLP, not ACL, and sorry for the confusion.", ["acknowledge_update", "acceptance_reassurance"], False, "synthetic"),
            ("No, the examples stay, but shorten the explanation.", ["acknowledge_update", "bounded_acknowledgment"], False, "synthetic"),
            ("Actually, that quote was Reviewer 2, and I can paste the full comment.", ["acknowledge_update", "accept_or_decline_offer"], False, "synthetic"),
            ("No, I meant the editor response, and thanks for pointing out the mismatch.", ["acknowledge_update", "acknowledgment"], False, "synthetic"),
            ("Actually, the dataset is public, not private, and please keep the reply concise.", ["acknowledge_update", "bounded_acknowledgment"], True, "synthetic"),
            ("No, that's the old title, and sorry for the confusion.", ["acknowledge_update", "acceptance_reassurance"], False, "synthetic"),
        ],
    },
    "frustration": {
        "expected": "repair_acknowledgment",
        "easy": [
            ("This is frustrating. You keep missing the main point.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("I'm frustrated because you're ignoring the format again.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("This is getting frustrating; you keep answering a different question.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("I'm frustrated that the reply keeps getting longer.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("This is frustrating; I already said no tables.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("I'm frustrated because the tone keeps drifting.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("This is frustrating. I already gave the word limit.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("I'm frustrated because you're still using bullets.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("This is frustrating; you keep sounding more certain after I correct you.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("I'm frustrated that I have to keep repeating the same constraints.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("This is frustrating because the answer keeps drifting off-topic.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("I'm frustrated that you're rewriting instead of answering directly.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("This is frustrating; I gave the venue and deadline already.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("I'm frustrated because the reply still sounds generic.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("This is frustrating — I corrected the year twice already.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("I'm frustrated because the style keeps changing between turns.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("This is frustrating; you keep missing the constraint hidden in the examples.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("I'm frustrated that you keep defaulting to long explanations.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
            ("This is frustrating; the last three responses all wandered.", ["repair_acknowledgment"], False, "adapted_from_real_conversation"),
            ("I'm frustrated because I keep having to re-specify the tone.", ["repair_acknowledgment"], True, "adapted_from_real_conversation"),
        ],
        "mixed_act": [
            ("This is frustrating, and please keep the next reply to one paragraph.", ["repair_acknowledgment", "bounded_acknowledgment"], True, "synthetic"),
            ("I'm frustrated, and I can paste the original constraints again if needed.", ["repair_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("This is frustrating; I already corrected the date twice.", ["repair_acknowledgment", "acknowledge_update"], True, "synthetic"),
            ("I'm frustrated because you keep ignoring the no-bullets rule, and please fix that now.", ["repair_acknowledgment", "bounded_acknowledgment"], True, "synthetic"),
            ("This is frustrating, and I can send the exact template again.", ["repair_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("I'm frustrated because you still use the old year after I corrected it.", ["repair_acknowledgment", "acknowledge_update"], True, "synthetic"),
            ("This is frustrating, and please just answer directly.", ["repair_acknowledgment", "bounded_acknowledgment"], False, "synthetic"),
            ("I'm frustrated, and sorry if I'm repeating myself.", ["repair_acknowledgment", "acceptance_reassurance"], False, "synthetic"),
            ("This is frustrating because the answer drifted, and I can resend the prompt.", ["repair_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("I'm frustrated that you're still not following the structure I corrected.", ["repair_acknowledgment", "acknowledge_update"], True, "synthetic"),
        ],
    },
    "preference": {
        "expected": "bounded_acknowledgment",
        "easy": [
            ("Please answer in plain prose only.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please keep the next reply under 100 words.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please stop using bold unless I ask.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
            ("Please give the answer first, then details only if needed.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
            ("Please use shorter sentences.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
            ("Please avoid bullet points.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please keep the tone neutral.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
            ("Please keep the response to one paragraph.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please answer directly without extra hedging.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please keep the explanation very brief.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
        ],
        "near_miss": [
            ("Please keep the same tone in the rest of this chat.", ["bounded_acknowledgment"], True, "synthetic"),
            ("Please don't switch back to bullets later.", ["bounded_acknowledgment"], True, "synthetic"),
            ("Please keep this concise style for the next reply too.", ["bounded_acknowledgment"], True, "synthetic"),
            ("Please use plain language and no extra caveats.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please answer in one paragraph and don't over-explain.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please keep this exact format from here on in this chat.", ["bounded_acknowledgment"], True, "synthetic"),
            ("Please keep the structure the same as before, but only here.", ["bounded_acknowledgment"], False, "synthetic"),
            ("Please stick to short direct answers unless I ask for more.", ["bounded_acknowledgment"], False, "adapted_from_real_conversation"),
            ("Please keep the same level of detail next turn.", ["bounded_acknowledgment"], True, "synthetic"),
            ("Please keep the tone calm and not too enthusiastic.", ["bounded_acknowledgment"], True, "adapted_from_real_conversation"),
        ],
        "mixed_act": [
            ("Please keep it short, and thanks for fixing the last one.", ["bounded_acknowledgment", "acknowledgment"], False, "synthetic"),
            ("Please use plain language, and I can send an example if helpful.", ["bounded_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Please answer directly, and sorry if I was vague earlier.", ["bounded_acknowledgment", "acceptance_reassurance"], True, "synthetic"),
            ("Please keep the structure the same, and thanks for the last revision.", ["bounded_acknowledgment", "acknowledgment"], False, "synthetic"),
            ("Please keep the response brief, and I can paste the full prompt again.", ["bounded_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Please avoid bullets, and sorry for changing the format again.", ["bounded_acknowledgment", "acceptance_reassurance"], True, "synthetic"),
            ("Please keep this tone, and I can share the exact sentence I liked.", ["bounded_acknowledgment", "accept_or_decline_offer"], False, "synthetic"),
            ("Please use one paragraph, and thanks for catching that mismatch.", ["bounded_acknowledgment", "acknowledgment"], False, "synthetic"),
            ("Please answer directly, and I corrected the year above.", ["bounded_acknowledgment", "acknowledge_update"], False, "synthetic"),
            ("Please keep it concise from here on, and sorry for the back-and-forth.", ["bounded_acknowledgment", "acceptance_reassurance"], True, "synthetic"),
        ],
    },
}


def build_records() -> list[dict]:
    records = []
    start_index = 21
    for family, config in FAMILY_CONFIGS.items():
        running_index = start_index
        for difficulty in ("easy", "near_miss", "mixed_act"):
            for idx, (user_turn, alts, risk, source_type) in enumerate(config[difficulty], start=1):
                records.append(
                    {
                        "id": f"{family}_{running_index:03d}",
                        "user_turn": user_turn,
                        "speech_act_family": family,
                        "expected_response_act": config["expected"],
                        "acceptable_alternative_response_acts": alts,
                        "difficulty": difficulty,
                        "context_notes": f"Auto-generated batch-3 benchmark item {idx} for {family} / {difficulty}.",
                        "gold_void_commitment_risk": risk,
                        "source_type": source_type,
                    }
                )
                running_index += 1
    return records


def main() -> None:
    existing_path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v2.jsonl"
    batch3_path = ROOT / "data" / "annotations" / "full_uptake_benchmark_batch3.jsonl"
    output_path = ROOT / "data" / "annotations" / "full_uptake_benchmark_v3.jsonl"

    existing = load_jsonl_records(existing_path)
    new_records = build_records()

    with batch3_path.open("w", encoding="utf-8") as handle:
        for record in new_records:
            handle.write(json.dumps(record) + "\n")

    with output_path.open("w", encoding="utf-8") as handle:
        for record in existing + new_records:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(new_records)} new items to {batch3_path.name}")
    print(f"Built {len(existing) + len(new_records)} items into {output_path.name}")


if __name__ == "__main__":
    main()
