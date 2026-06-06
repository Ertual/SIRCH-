import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.decision import DecisionEngine, decide, reset_decision_buffer
from core.preprocessing import FrameSequenceBuffer, build_sequence
from database.db import add_incident, get_settings, list_incidents, save_settings
from alerts.alertmanager import AlertManager, AlertPayload


class DecisionTests(unittest.TestCase):
    def test_function_decide_waits_for_full_window(self):
        reset_decision_buffer()
        self.assertFalse(decide(1.0))
        self.assertFalse(decide(1.0))
        self.assertFalse(decide(1.0))
        self.assertFalse(decide(1.0))
        self.assertTrue(decide(1.0))

    def test_decision_engine_uses_mean_score(self):
        engine = DecisionEngine(threshold=0.5, k_window=3)
        self.assertEqual(engine.update(0.9)[0], False)
        self.assertEqual(engine.update(0.1)[0], False)
        decision, mean_score = engine.update(0.8)
        self.assertTrue(decision)
        self.assertAlmostEqual(mean_score, 0.6)


class PreprocessingTests(unittest.TestCase):
    def test_frame_sequence_buffer_returns_expected_shape(self):
        buffer = FrameSequenceBuffer()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        sequence = None
        for _ in range(20):
            sequence = buffer.add_frame(frame)
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.shape, (1, 20, 224, 224, 3))

    def test_build_sequence_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            build_sequence([])


class DatabaseTests(unittest.TestCase):
    def test_incident_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "sirch.db")
            incident_id = add_incident(0.7, 0.6, "capture.jpg", "test", db_path=db_path)
            rows = list_incidents(db_path=db_path)
            self.assertEqual(incident_id, 1)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["score"], 0.7)

    def test_settings_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "sirch.db")
            save_settings(0.65, 7, db_path=db_path)
            settings = get_settings(db_path=db_path)
            self.assertEqual(settings["threshold"], 0.65)
            self.assertEqual(settings["k_window"], 7)


class AlertTests(unittest.TestCase):
    def test_telegram_placeholder_is_ignored(self):
        manager = AlertManager()
        manager.send_telegram(AlertPayload(score=0.9))


if __name__ == "__main__":
    unittest.main()
