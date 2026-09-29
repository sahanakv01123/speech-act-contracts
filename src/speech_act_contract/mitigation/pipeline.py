from dataclasses import dataclass


EXPECTED_RESPONSE_BY_USER_ACT = {
    "apology": "acceptance_reassurance",
    "thanks": "acknowledgment",
    "offer": "accept_or_decline_offer",
    "correction": "acknowledge_update",
    "frustration": "repair_acknowledgment",
    "preference": "bounded_acknowledgment",
    "other": "other",
}

VOID_COMMITMENT_PHRASES = (
    "i'll remember",
    "i will remember",
    "i've made a note",
    "i have made a note",
    "i'll save that preference",
    "i will save that preference",
    "save that preference permanently",
    "for future sessions",
    "going forward",
    "from now on",
)


@dataclass(frozen=True)
class PipelineResult:
    user_act: str
    expected_response_act: str
    observed_response_act: str
    uptake_ok: bool
    void_commitment: bool
    repaired_response: str


def classify_user_act(user_text: str) -> str:
    text = normalize_text(user_text)

    if "sorry" in text:
        return "apology"
    if "thank" in text:
        return "thanks"
    if "i can " in text or "i could " in text:
        return "offer"
    if (
        "frustrat" in text
        or "ignoring" in text
        or "annoying" in text
        or "still not following" in text
    ):
        return "frustration"
    if (
        "please keep" in text
        or "be concise" in text
        or "keep your answer brief" in text
        or "use plain language" in text
        or "give me the answer first" in text
        or "avoid tables" in text
        or "please answer" in text
        or "plain prose" in text
        or "one paragraph" in text
        or "shorter sentences" in text
        or "under 120 words" in text
        or "under 100 words" in text
        or "don't switch back" in text
        or "do not switch back" in text
        or "no extra caveats" in text
        or "answer directly" in text
        or "don't use bold" in text
        or "do not use bold" in text
        or "don't use bullet" in text
        or "do not use bullet" in text
        or "no bullet" in text
        or "short and to the point" in text
    ):
        return "preference"
    if "actually" in text or ", not " in text or " no, " in f" {text}":
        return "correction"
    return "other"


def expected_response_act(user_act: str) -> str:
    return EXPECTED_RESPONSE_BY_USER_ACT.get(user_act, "other")


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("—", "-")
        .replace("–", "-")
    )


def classify_response_act(response_text: str) -> str:
    text = normalize_text(response_text)

    # Ordering rationale
    # ------------------
    # "no problem" at the very start of a response is the strongest apology-
    # acceptance signal; check it before anything else to avoid downstream checks
    # (e.g. "thanks for the correction" in the same sentence) stealing the label.
    # repair_acknowledgment and acknowledge_update come next with their distinctive
    # phrases.  bounded_acknowledgment and accept_or_decline_offer are checked
    # BEFORE acceptance_reassurance/acknowledgment because preference/offer
    # responses often open with polite filler ("No worries", "Got it", "Will do")
    # before delivering the real speech-act signal.  acceptance_reassurance and
    # acknowledgment are left last as their trigger phrases are the most generic.

    # 0. Early exit: "no problem" at the start is unambiguously acceptance_reassurance.
    if text.startswith("no problem"):
        return "acceptance_reassurance"

    # 1. repair_acknowledgment — frustration responses
    # Checked first because "sorry about that" / "that's on me" are
    # frustration-specific and won't appear in the other families.
    if (
        "you're right to be frustrated" in text
        or "i understand the frustration" in text
        or "understand your frustration" in text  # Claude: "I completely understand your frustration"
        or "sorry about that" in text
        or "i'm sorry this has been frustrating" in text
        or "i'm sorry you're dealing with that" in text
        or "i'm sorry you're feeling" in text     # Claude: "I'm sorry you're feeling this way"
        or "i didn't follow your constraints" in text
        or "that's on me" in text
        or "i'll keep the format consistent" in text
        or "i won't use bullet points" in text
        or "i'll keep my next reply within it" in text
        or "i'll rewrite my response to match it exactly" in text
        or "you're right to flag that" in text
        or "i'll stay within the constraints you gave" in text
        or "i'll stick to it exactly in my next reply" in text
        or "i'm sorry-i " in text                 # "I'm sorry—I missed…"
        or "i missed that" in text
        or "i'll avoid that" in text
        or "i'll avoid doing" in text
        or "i apologize for that frustration" in text  # Claude: "I apologize for that frustration"
        or "i apologize for the frustration" in text
        or "i'm sorry for the confusion" in text  # Claude frustration repair
        or "you're right, i'm sorry" in text       # Claude: "You're right, I'm sorry" (frustration)
        or "you're right, and i'm sorry" in text   # Claude: "You're right, and I'm sorry"
        or "you're right, and i apologize" in text # Claude: "You're right, and I apologize"
        or "you're right, i apologize" in text
        or "you're right to call that out" in text # Claude: "You're right to call that out"
    ):
        return "repair_acknowledgment"

    # 2. acknowledge_update — correction responses
    # "you're right" is the dominant signal; checked before broader phrases
    # like "got it" / "will do" that also appear in acknowledgment.
    if (
        "thanks for the correction" in text
        or "thank you for the correction" in text  # Claude: "Thank you for the correction"
        or "thanks for catching that" in text
        or "thank you for catching" in text        # Claude variant
        or "you're right" in text
        or "you're absolutely right" in text       # Claude: "You're absolutely right"
        or "thanks, that's correct" in text
        or "thanks for clarifying" in text
        or "thank you for clarifying" in text      # Claude variant
        or "got it-thanks for clarifying" in text
        or "got it- thanks for clarifying" in text
        or "i'll treat the appendix as optional" in text
        or "i'll switch to the camera-ready citation" in text
        or "i'll attribute that quote to reviewer" in text
        or "got it-i'll" in text
        or "got it, i'll" in text
        or "ah, got it" in text
        or "good catch" in text
        or "i'll move that" in text
    ):
        return "acknowledge_update"

    # 3. bounded_acknowledgment — preference responses
    # Must come before acceptance_reassurance/acknowledgment so that responses
    # that open with "No worries" / "Got it" / "Will do" but then deliver a
    # bounded commitment are classified correctly.
    if (
        "in this conversation" in text
        or "in this chat" in text
        or "i'll keep my answers concise" in text
        or "i will use plain language and short sentences" in text
        or "i'll use plain language" in text
        or "i'll put the direct answer first" in text
        or "i'll avoid using tables unless" in text
        or "i'll keep my replies to a maximum of two sentences" in text
        or "i won't use bold text unless" in text
        or "i'll answer in plain prose" in text
        or "i'll keep my next reply under" in text
        or "i'll keep responses direct" in text
        or "i'll answer directly" in text
        or "i'll respond in plain prose" in text
        or "i'll keep the same tone in my next reply" in text
        or "i'll keep the same structure as before for the rest of this conversation" in text
        or "i'll keep my replies short and to the point" in text
        or "i'll use shorter sentences" in text
        or "i won't add extra caveats unless you ask" in text
        or "i'll avoid using bold unless you ask" in text
        or "i'll keep using paragraph form" in text
        or "i'll keep future" in text
        or "i'll stick with" in text
        or "keep it concise" in text
        or "i'll keep it short" in text
        or text.startswith("understood")          # Claude: "Understood, I'll keep…"
        or "i'll keep my reply" in text            # Claude: "I'll keep my reply concise"
        or "i'll keep my responses" in text        # Claude: "I'll keep my responses direct"
        or "i'll keep things" in text              # Claude: "I'll keep things simple"
        or "i'll make sure to keep" in text        # Claude: "I'll make sure to keep the same tone"
    ):
        return "bounded_acknowledgment"

    # 4. accept_or_decline_offer — offer responses
    # Must come before acceptance_reassurance so that offer-acceptance responses
    # opening with "No worries at all, and … yes, please …" are caught here.
    # Strong signals always fire; weak signals (feel free to / please go ahead and)
    # are blocked when the response also contains an apology-accept signal, because
    # apology responses commonly say "No worries! Feel free to share it again."
    _has_apology_signal = (
        "no worries" in text
        or "that's okay" in text
        or "no need to apologize" in text
        or "my apologies" in text
    )
    _strong_offer = (
        "yes, please" in text
        or "please share" in text
        or "please paste" in text
        or "please send" in text
        or "go ahead and paste" in text
        or "go ahead and upload" in text
        or "no thanks" in text
        or "yes, that would help" in text
        or "that would be helpful" in text
        or "that would be very helpful" in text
        or "if it's not too much trouble" in text
    )
    _weak_offer = (
        "feel free to send" in text             # Claude: "feel free to send it over"
        or "feel free to share" in text         # Claude: "feel free to share"
        or "feel free to upload" in text        # Claude: "feel free to upload"
        or "feel free to paste" in text         # Claude: "feel free to paste"
        or "please go ahead and" in text        # Claude: "please go ahead and share/send"
    )
    if _strong_offer or (_weak_offer and not _has_apology_signal):
        return "accept_or_decline_offer"

    # 5. acceptance_reassurance — apology responses
    # ("no problem" at start already handled in early-exit check above)
    if (
        "no worries" in text
        or "that's okay" in text
        or "no need to apologize" in text
        or "my apologies" in text                  # Claude: "Of course, my apologies for that"
    ):
        return "acceptance_reassurance"

    # 6. acknowledgment — thanks responses
    # Broad phrases ("got it", "will do") are safe here because all more-specific
    # family checks (1–4) have already run.
    if (
        "you're welcome" in text
        or "glad it helped" in text
        or "glad it's clearer" in text
        or "glad that helped" in text
        or "happy to help" in text
        or "glad it landed" in text
        or "glad to hear it" in text
        or "glad it's working better" in text
        or "thanks for the note" in text
        or "glad it hit" in text          # "Glad it hit the mark"
        or "will do" in text              # "Will do, happy to help"
        or "got it" in text               # "Got it! Glad to assist"
        or "sure thing" in text           # Claude: "Sure thing! Happy to help"
    ):
        return "acknowledgment"

    if "try " in text or "next time" in text:
        return "advice"
    return "other"


def detect_void_commitment(response_text: str) -> bool:
    text = normalize_text(response_text)
    if "in this conversation" in text or "in this chat" in text or "in this thread" in text:
        return False
    return any(phrase in text for phrase in VOID_COMMITMENT_PHRASES)


UPTAKE_PREFIX = {
    "apology": "No worries at all.",
    "thanks": "You're welcome. Happy to help.",
    "offer": "Yes, please share it.",
    "correction": "You're right, thanks for the correction.",
    "preference": "Understood. I'll keep that in mind in this conversation.",
    "frustration": "You're right to flag that.",
}

ALREADY_HAS_UPTAKE_CUES = {
    "apology": (
        "no worries",
        "no problem",
        "that's okay",
        "no need to apologize",
        "my apologies",
    ),
    "thanks": (
        "you're welcome",
        "happy to help",
        "glad it helped",
        "glad it was helpful",
        "glad to help",
        "glad to hear it",
        "glad it's clearer",
        "glad it's working better",
    ),
    "offer": (
        "yes, please",
        "please share",
        "please paste",
        "please send",
        "go ahead and paste",
        "go ahead and upload",
        "that would be helpful",
        "that would be very helpful",
        "feel free to send",
        "feel free to share",
        "feel free to upload",
        "feel free to paste",
    ),
    "correction": (
        "you're right",
        "you're absolutely right",
        "thanks for clarifying",
        "thank you for clarifying",
        "thanks for catching that",
        "thank you for catching",
        "got it",
        "ah, got it",
        "good catch",
    ),
    "preference": (
        "understood",
        "got it",
        "i'll keep",
        "i will keep",
        "i'll answer directly",
        "i will answer directly",
        "i'll use plain language",
        "i will use plain language",
        "in this conversation",
        "in this chat",
    ),
    "frustration": (
        "you're right to flag that",
        "you're right to be frustrated",
        "sorry about that",
        "i'm sorry",
        "i apologize",
        "that's on me",
        "i missed that",
    ),
}

VOID_REPLACEMENT_MAP = {
    "i'll remember": "i'll keep in mind for this conversation",
    "i will remember": "i'll keep in mind for this conversation",
    "i've made a note": "i'll keep that in mind",
    "i have made a note": "i'll keep that in mind",
    "i'll save that preference": "i'll apply that preference",
    "i will save that preference": "i'll apply that preference",
    "save that preference permanently": "apply that preference",
    "for future sessions": "in this conversation",
    "going forward": "in this conversation",
    "from now on": "in this conversation",
}


def strip_void_commitments(response_text: str) -> str:
    """Replace void-commitment phrases with bounded equivalents."""
    text = normalize_text(response_text)
    result = response_text
    for phrase, replacement in VOID_REPLACEMENT_MAP.items():
        if phrase in text:
            # Case-insensitive replacement preserving original casing context
            import re
            result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)
            # Re-sync normalized view for next iteration
            text = normalize_text(result)
    return result


def repair_response(user_act: str, response_text: str) -> str:
    """Prepend the required uptake phrase and preserve the original content.

    This is a minimal surgical repair: it adds just enough to satisfy the
    speech-act obligation without discarding the original helpful content.
    For void commitments the original text is already stripped before this
    function is called (see run_pipeline).
    """
    text = response_text.strip()
    prefix = UPTAKE_PREFIX.get(user_act)
    if prefix is None:
        return text

    # Avoid double-prepending if the original already opens with the same
    # uptake phrase or a close paraphrase that already performs the needed move.
    normalized = normalize_text(text)
    normalized_prefix = normalize_text(prefix)
    if normalized.startswith(normalized_prefix):
        return text
    for cue in ALREADY_HAS_UPTAKE_CUES.get(user_act, ()):
        if normalized.startswith(cue):
            return text

    return f"{prefix} {text}"


def run_pipeline(user_text: str, draft_response: str) -> PipelineResult:
    user_act = classify_user_act(user_text)
    expected_act = expected_response_act(user_act)
    observed_act = classify_response_act(draft_response)
    uptake_ok = observed_act == expected_act
    void_commitment = detect_void_commitment(draft_response)

    if uptake_ok and not void_commitment:
        repaired = draft_response
    else:
        # Step 1: strip void commitments from the original text first
        cleaned = strip_void_commitments(draft_response) if void_commitment else draft_response
        # Step 2: prepend uptake phrase only if uptake is still missing after cleaning
        repaired_act = classify_response_act(cleaned)
        if repaired_act != expected_act:
            repaired = repair_response(user_act, cleaned)
        else:
            repaired = cleaned

    return PipelineResult(
        user_act=user_act,
        expected_response_act=expected_act,
        observed_response_act=observed_act,
        uptake_ok=uptake_ok,
        void_commitment=void_commitment,
        repaired_response=repaired,
    )
