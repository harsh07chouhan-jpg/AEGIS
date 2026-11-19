import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_inference_service import AudioInferenceService


VALIDATION_DIR = ROOT / "results" / "live_validation_100"

MANIFEST = VALIDATION_DIR / "blinded_upload_list.csv"
OUTPUT = VALIDATION_DIR / "batch_predictions.jsonl"

EXPECTED_FIRST = "LA_D_5702258.flac"
EXPECTED_SECOND = "LA_D_6035951.flac"
EXPECTED_CASES = 100


def main():
    print("=" * 65)
    print("AEGIS - CONTROLLED LIVE VALIDATION BATCH")
    print("=" * 65)

    # Safety checks: never overwrite an existing result.
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT}"
        )

    # Read only the blinded manifest. Never open the ground-truth key.
    with MANIFEST.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"Expected 100 cases, found {len(rows)}")

    expected_ids = [f"AEGIS-{i:03d}" for i in range(1, 101)]
    actual_ids = [row["case_id"] for row in rows]

    if actual_ids != expected_ids:
        raise ValueError("Case IDs are missing, duplicated, or out of order.")

    if Path(rows[0]["audio_file"]).name != EXPECTED_FIRST:
        raise ValueError("AEGIS-001 does not match the verified recording.")

    if Path(rows[1]["audio_file"]).name != EXPECTED_SECOND:
        raise ValueError("AEGIS-002 does not match the verified recording.")

    # The first two cases were already tested manually.
    remaining = rows[2:]

    missing = [
        row["audio_file"]
        for row in remaining
        if not Path(row["audio_file"]).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files are missing. First: {missing[0]}"
        )

    print(f"Manifest verified: {len(rows)} cases")
    print("Manual cases preserved: AEGIS-001 and AEGIS-002")
    print(f"Automated cases: {len(remaining)}")
    print("Ground-truth key: NOT OPENED")
    print("\nLoading the existing verified inference service...")

    # Same model, preprocessing, inference method and threshold as the API.
    service = AudioInferenceService(num_threads=2)

    print("Inference service loaded successfully.")
    print(f"Saving predictions to:\n{OUTPUT}\n")

    completed = 0
    errors = 0

    # Exclusive creation prevents accidental overwriting.
    with OUTPUT.open("x", encoding="utf-8") as output_file:
        for index, row in enumerate(remaining, start=3):
            case_id = row["case_id"]
            audio_path = Path(row["audio_file"])

            print(
                f"[{index - 2:02d}/{len(remaining)}] "
                f"Processing {case_id}: {audio_path.name}"
            )

            try:
                result = service.analyze_file(
                    audio_path,
                    display_name=case_id,
                )

                status = result.get("status", "unknown")

                record = {
                    "case_id": case_id,
                    "audio_file": str(audio_path),
                    "status": status,
                    "result": result,
                }

                if status == "ok":
                    completed += 1
                    decision = result.get("decision", {}).get(
                        "label", "unknown"
                    )
                    score = result.get("aggregate", {}).get(
                        "mean_spoof_score"
                    )
                    print(
                        f"    Result: {decision} | "
                        f"Mean spoof score: {score}"
                    )
                else:
                    errors += 1
                    print(f"    Inference status: {status}")

            except Exception as exc:
                errors += 1
                record = {
                    "case_id": case_id,
                    "audio_file": str(audio_path),
                    "status": "batch_error",
                    "error": repr(exc),
                }
                print(f"    ERROR: {exc}")

            # Save every case immediately to protect completed results.
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()

    print("\n" + "=" * 65)
    print("BATCH PROCESSING FINISHED")
    print("=" * 65)
    print(f"Successfully analysed: {completed}")
    print(f"Errors/rejections:     {errors}")
    print(f"Total processed:       {completed + errors}")
    print(f"Output file:           {OUTPUT}")
    print("\nGround-truth key remains unopened.")
    print("Do not calculate accuracy until all 100 predictions are assembled.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nBATCH STOPPED SAFELY: {exc}")
        sys.exit(1)
