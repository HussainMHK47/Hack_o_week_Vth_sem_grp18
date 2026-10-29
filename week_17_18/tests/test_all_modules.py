"""
Automated Test Suite for AgroPulse AI
SIT-N Hack-O-Week (Weeks 17 & 18)

Verifies all 7 Deep Learning & Agronomy diagnostic modules.
"""

import sys
import os
import unittest
import torch

# Ensure all module roots are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_DIRS = [
    os.path.join(BASE_DIR, "01_tensors_and_autograd"),
    os.path.join(BASE_DIR, "02_cnn_classification"),
    os.path.join(BASE_DIR, "03_sequential_rnn_lstm_gru"),
    os.path.join(BASE_DIR, "04_regularization_techniques"),
    os.path.join(BASE_DIR, "05_transfer_learning_cnn"),
    os.path.join(BASE_DIR, "06_transformers_nlp"),
    os.path.join(BASE_DIR, "07_llm_agronomy_copilot")
]
for p in MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class TestAgroPulseModules(unittest.TestCase):

    def test_01_tensors_and_autograd(self):
        """Test Module 1: Tensors, autograd, and non-linear yield optimizer."""
        from crop_yield_optimizer import CropYieldModel, generate_synthetic_yield_data
        N_levels, Y_obs, true_p = generate_synthetic_yield_data(num_samples=20)
        self.assertEqual(len(N_levels), 20)
        self.assertEqual(len(Y_obs), 20)

        model = CropYieldModel()
        preds = model(N_levels)
        loss = torch.mean((preds - Y_obs) ** 2)
        loss.backward()

        self.assertIsNotNone(model.A.grad)
        self.assertIsNotNone(model.k.grad)
        self.assertIsNotNone(model.c.grad)
        self.assertGreater(abs(model.A.grad.item()), 0.0)

    def test_02_cnn_classification(self):
        """Test Module 2: AgriLeafNet architecture and feature map extractors."""
        from cnn_model import AgriLeafNet
        from dataset_generator import generate_single_leaf_image, CLASS_NAMES

        # Image generation test
        for cname in CLASS_NAMES:
            img = generate_single_leaf_image(cname, img_size=32)
            self.assertEqual(img.size, (32, 32))

        # CNN forward pass test
        model = AgriLeafNet(num_classes=4)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 4))

        # Feature map extraction test
        fmaps = model.extract_feature_maps(x)
        self.assertIn("block1", fmaps)
        self.assertIn("block2", fmaps)
        self.assertIn("block3", fmaps)
        self.assertEqual(fmaps["block1"].shape, (2, 32, 32, 32))

    def test_03_sequential_models(self):
        """Test Module 3: Vanilla RNN, LSTM, and GRU architectures & sliding window dataset."""
        from models import build_sequential_model
        from telemetry_dataset import generate_iot_telemetry_series, TelemetrySequenceDataset

        df = generate_iot_telemetry_series(num_hours=72)
        self.assertEqual(len(df), 72)

        feat_arr = df[["soil_moisture_pct", "air_temperature_c", "relative_humidity_pct", "solar_radiation_wm2", "soil_ph"]].values
        ds = TelemetrySequenceDataset(feat_arr, seq_len=12)
        x, y = ds[0]
        self.assertEqual(x.shape, (12, 5))
        self.assertEqual(y.shape, (1,))

        # Models test
        dummy_input = torch.randn(4, 12, 5)
        for mname in ["Vanilla_RNN", "LSTM", "GRU"]:
            seq_m = build_sequential_model(mname, input_size=5, hidden_dim=16, num_layers=1)
            out = seq_m(dummy_input)
            self.assertEqual(out.shape, (4, 1))

    def test_04_regularization(self):
        """Test Module 4: Baseline vs Dropout vs BatchNorm and EarlyStopping."""
        from regularization_models import build_regularization_variant, EarlyStopping

        xb = torch.randn(8, 12)
        for v in ["Baseline", "Dropout", "BatchNorm", "Full"]:
            m = build_regularization_variant(v, in_features=12, num_classes=4)
            m.eval()
            out = m(xb)
            self.assertEqual(out.shape, (8, 4))

        # Early stopping test
        es = EarlyStopping(patience=2, min_delta=0.01)
        dummy_net = torch.nn.Linear(2, 2)
        stop1 = es.step(1.5, dummy_net, epoch=1)
        self.assertFalse(stop1)
        stop2 = es.step(1.6, dummy_net, epoch=2)
        self.assertFalse(stop2)
        stop3 = es.step(1.7, dummy_net, epoch=3)
        self.assertTrue(stop3)
        self.assertEqual(es.stopped_epoch, 3)

    def test_05_transfer_learning(self):
        """Test Module 5: Pretrained MobileNetV2 and ResNet18 parameter freezing."""
        from transfer_models import build_transfer_model, get_trainable_param_stats

        m_frozen = build_transfer_model("mobilenet_v2", num_classes=4, mode="feature_extract", pretrained=False)
        stats_frozen = get_trainable_param_stats(m_frozen)
        self.assertLess(stats_frozen["trainable_pct"], 15.0)

        m_scratch = build_transfer_model("mobilenet_v2", num_classes=4, mode="from_scratch", pretrained=False)
        stats_scratch = get_trainable_param_stats(m_scratch)
        self.assertEqual(stats_scratch["trainable_pct"], 100.0)

    def test_06_transformers_nlp(self):
        """Test Module 6: Agronomy NLP Triage intent and entity extraction."""
        from nlp_triage import AgronomyNLPTriageAgent

        triage = AgronomyNLPTriageAgent(use_hf_pipeline=False)
        res = triage.analyze_inquiry("Tomato leaves have orange rust pustules after heavy rain.")
        self.assertIn("Pathology_Diagnosis", res["all_intent_scores"])
        self.assertIn("Tomato", res["extracted_entities"]["crops"])
        self.assertIn("rust", res["extracted_entities"]["detected_symptoms"])

    def test_07_llm_copilot(self):
        """Test Module 7: Multi-modal fusion copilot with deterministic fallback."""
        from diagnostic_agent import MultiModalAgronomyCopilot

        copilot = MultiModalAgronomyCopilot()
        report = copilot.generate_advisory_report(
            vision_data={"predicted_class": "Cereal_Rust", "confidence_pct": 97.5},
            telemetry_data={"current_moisture_pct": 32.0, "air_temp_c": 24.0},
            nlp_data={"raw_query": "Wheat crop orange rust", "urgency_level": "ELEVATED"}
        )
        self.assertIn("report_markdown", report)
        self.assertIn("Cereal Rust", report["report_markdown"])
        self.assertIn("Remediation Protocol", report["report_markdown"])


def run_tests():
    print("=" * 65)
    print("AGROPULSE AI - AUTOMATED VERIFICATION TEST SUITE")
    print("SIT-N Hack-O-Week (Weeks 17 & 18)")
    print("=" * 65 + "\n")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgroPulseModules)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    if result.wasSuccessful():
        print(f"[TEST SUMMARY] ALL {result.testsRun} DEEP LEARNING MODULE TESTS PASSED (100% SUCCESS)!")
    else:
        print(f"[TEST SUMMARY] Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print("=" * 65)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
