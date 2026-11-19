"""AEGIS Backend and Inference Integration Test Suite.

Validates:
1. Checkpoint SHA-256 integrity and evaluation summary parity.
2. FastAPI endpoints: GET /, GET /health, GET /model-info.
3. Audio analysis regression using verified test utterance LA_E_2834763.flac.
4. Input rejection for empty, invalid, and corrupt audio files.
5. Temporary file cleanup guarantees.
"""

import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
from api import app, service_state, MAX_UPLOAD_BYTES
from audio_inference_service import (
    AudioInferenceService,
    EXPECTED_CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
)

TEST_FLAC_PATH = (
    PROJECT_ROOT
    / "data/audio_dev/raw/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac"
)


class TestAegisCheckpointAndMetadata(unittest.TestCase):
    """Integrity checks on models and recorded evaluation benchmarks."""

    def test_checkpoint_sha256_integrity(self):
        self.assertTrue(
            DEFAULT_CHECKPOINT_PATH.is_file(),
            f"Checkpoint file missing: {DEFAULT_CHECKPOINT_PATH}",
        )
        digest = hashlib.sha256(DEFAULT_CHECKPOINT_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(
            digest,
            EXPECTED_CHECKPOINT_SHA256,
            "Checkpoint SHA-256 does not match authoritative verified hash!",
        )

    def test_evaluation_summary_parity(self):
        self.assertTrue(
            DEFAULT_EVAL_SUMMARY_PATH.is_file(),
            f"Evaluation summary missing: {DEFAULT_EVAL_SUMMARY_PATH}",
        )
        data = json.loads(DEFAULT_EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            data.get("checkpoint_sha256", "").upper(),
            EXPECTED_CHECKPOINT_SHA256,
        )
        self.assertEqual(data.get("metadata_utterances"), 71237)
        self.assertEqual(data.get("prediction_rows"), 71237)
        self.assertAlmostEqual(data.get("accuracy", 0), 0.8903519, places=4)
        self.assertAlmostEqual(data.get("balanced_accuracy", 0), 0.7473935, places=4)
        self.assertAlmostEqual(data.get("roc_auc", 0), 0.8845935, places=4)
        self.assertAlmostEqual(data.get("eer_approx", 0), 0.1908947, places=4)


class TestAegisApiEndpoints(unittest.TestCase):
    """End-to-end API tests using FastAPI TestClient with lifespan initialization."""

    @classmethod
    def setUpClass(cls):
        # Enter FastAPI lifespan so service is initialized
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("name"), "AEGIS Audio Forensics API")
        self.assertEqual(data.get("status"), "ready")
        self.assertEqual(data.get("docs"), "/docs")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertIsNone(data.get("startup_error"))

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("schema_version"), "aegis.audio_inference.v1")

        model = data.get("model", {})
        self.assertEqual(model.get("name"), "aegis_audio_transformer_reproducible_v1")
        self.assertEqual(model.get("checkpoint_sha256"), EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(model.get("checkpoint_epoch"), 5)
        self.assertEqual(model.get("parameter_count"), 875906)

        policy = data.get("policy", {})
        self.assertEqual(policy.get("sample_rate"), 16000)
        self.assertEqual(policy.get("decision_threshold"), 0.5)

        ref = data.get("reference_performance", {})
        self.assertEqual(ref.get("utterances"), 71237)
        self.assertIn("limitations", data)
        self.assertGreater(len(data["limitations"]), 0)

    def test_analyze_audio_known_sample_regression(self):
        self.assertTrue(
            TEST_FLAC_PATH.is_file(),
            f"Regression audio test file not found: {TEST_FLAC_PATH}",
        )
        with open(TEST_FLAC_PATH, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/analyze/audio",
            files={"file": ("LA_E_2834763.flac", file_bytes, "audio/flac")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("status"), "ok")
        input_meta = data.get("input", {})
        self.assertEqual(input_meta.get("filename"), "LA_E_2834763.flac")
        self.assertAlmostEqual(input_meta.get("duration_sec", 0), 1.426937, places=3)
        expected_file_sha = hashlib.sha256(file_bytes).hexdigest().upper()
        self.assertEqual(input_meta.get("input_sha256"), expected_file_sha)

        aggregate = data.get("aggregate", {})
        self.assertEqual(aggregate.get("num_windows"), 1)
        mean_score = aggregate.get("mean_spoof_score", 0)
        max_score = aggregate.get("max_spoof_score", 0)
        self.assertAlmostEqual(mean_score, 0.7161, places=3)
        self.assertAlmostEqual(max_score, 0.7161, places=3)

        decision = data.get("decision", {})
        self.assertEqual(decision.get("label"), "spoof")
        self.assertIn("0.5", decision.get("rule", ""))

        warnings = data.get("warnings", [])
        warning_codes = [w.get("code") for w in warnings]
        self.assertIn("zero_padded_window", warning_codes)

        windows = data.get("windows", [])
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].get("spoof_score", 0), 0.7161, places=3)

    def test_analyze_empty_file_rejected(self):
        response = self.client.post(
            "/analyze/audio",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json().get("detail", {})
        self.assertEqual(detail.get("code"), "empty_upload")

    def test_analyze_corrupt_file_rejected(self):
        # Corrupt pseudo-audio payload
        corrupt_data = b"RIFF" + b"\x00" * 32
        response = self.client.post(
            "/analyze/audio",
            files={"file": ("corrupt.wav", corrupt_data, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "rejected")
        self.assertIn("code", data.get("error", {}))

    def test_temporary_files_are_cleaned_up(self):
        import tempfile
        temp_dir = Path(tempfile.gettempdir())

        # Ingest an audio file
        with open(TEST_FLAC_PATH, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/analyze/audio",
            files={"file": ("cleanup_test.flac", file_bytes, "audio/flac")},
        )
        self.assertEqual(response.status_code, 200)

        # Check that no temporary files prefixed with aegis_upload_ remain
        leaked_files = list(temp_dir.glob("aegis_upload_*"))
        self.assertEqual(
            len(leaked_files),
            0,
            f"Temporary upload files were leaked in temp directory: {leaked_files}",
        )


if __name__ == "__main__":
    unittest.main()
