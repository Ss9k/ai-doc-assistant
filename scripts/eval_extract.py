import json
import time
import urllib.error
import urllib.request
from json import JSONDecodeError
from pathlib import Path

DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "resumes" / "expected.json"
)

API_URL = "http://127.0.0.1:8000/extract"

PAUSE_SECONDS = 2


def call_extract_api(resume_text: str) -> dict:
    """Send resume text to the extraction API."""

    payload = json.dumps({"resume_text": resume_text}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate():
    """Evaluate extraction results against expected outputs."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    resumes = dataset["resumes"]

    total_resumes = len(resumes)
    valid_responses = 0
    correct_fields = 0
    total_fields = total_resumes * 4

    print(f"\nStarting resume evaluation: {total_resumes} resumes\n")

    for index, item in enumerate(resumes, start=1):
        resume_id = item["id"]
        resume_text = item["resume_text"]
        expected = item["expected"]

        print(f"[{index}/{total_resumes}] Evaluating Resume {resume_id}...")

        try:
            actual = call_extract_api(resume_text)
            required_fields = {"name", "email", "phone", "skills"}

            if not isinstance(actual, dict) or not required_fields.issubset(actual):
                raise ValueError("API returned an invalid response schema.")

            if not (
                (actual["name"] is None or isinstance(actual["name"], str))
                and (actual["email"] is None or isinstance(actual["email"], str))
                and (actual["phone"] is None or isinstance(actual["phone"], str))
                and isinstance(actual["skills"], list)
                and all(isinstance(skill, str) for skill in actual["skills"])
            ):
                raise ValueError("API response contains invalid field types.")

            valid_responses += 1
            resume_correct_fields = 0

            for field in ["name", "email", "phone", "skills"]:
                actual_value = actual.get(field)
                expected_value = expected.get(field)
                if actual.get(field) == expected.get(field):
                    correct_fields += 1
                    resume_correct_fields += 1
                else:
                    print(f"  MISMATCH: {field}")
                    print(f"    Expected: {expected_value}")
                    print(f"    Actual:   {actual_value}")

            print(f"  Valid response | Fields correct: {resume_correct_fields}/4")

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            print(f"  API error: HTTP {error.code} - {error_body}")

        except (urllib.error.URLError, TimeoutError, JSONDecodeError) as error:
            print(f"  Evaluation error: {error}")

        if index < total_resumes:
            time.sleep(PAUSE_SECONDS)

    accuracy = (correct_fields / total_fields) * 100 if total_fields else 0

    print("\n" + "=" * 45)
    print("          EVALUATION SUMMARY")
    print("=" * 45)

    print(f"Total resumes:       {total_resumes}")
    print(f"Valid responses:     {valid_responses}/{total_resumes}")
    print(f"Correct fields:      {correct_fields}/{total_fields}")
    print(f"Field accuracy:      {accuracy:.2f}%")

    print("=" * 45)


if __name__ == "__main__":
    evaluate()
