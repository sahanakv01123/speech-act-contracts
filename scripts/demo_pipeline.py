from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_act_contract.mitigation.pipeline import run_pipeline


def main() -> None:
    user_text = "Sorry I wasn't clear earlier. Please keep your answer brief from now on."
    draft_response = "I will remember to keep future answers brief. Next time, structure your request more clearly."

    result = run_pipeline(user_text, draft_response)

    print("USER TURN:")
    print(user_text)
    print()
    print("RAW DRAFT:")
    print(draft_response)
    print()
    print("PIPELINE RESULT:")
    print(f"user_act={result.user_act}")
    print(f"expected_response_act={result.expected_response_act}")
    print(f"observed_response_act={result.observed_response_act}")
    print(f"uptake_ok={result.uptake_ok}")
    print(f"void_commitment={result.void_commitment}")
    print()
    print("REPAIRED RESPONSE:")
    print(result.repaired_response)


if __name__ == "__main__":
    main()
