"""用可手算行情检查时序对齐与截面样本门槛"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from experiment import calculate, main
from quanttoolkit.tracelog import configure_log, print_call_chain


class MomentumExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_directory = tempfile.TemporaryDirectory()
        cls.log_path = Path(cls.log_directory.name) / "trace.log"
        configure_log(cls.log_path)

    @classmethod
    def tearDownClass(cls):
        with redirect_stdout(io.StringIO()):
            print_call_chain()
        cls.log_directory.cleanup()

    def setUp(self):
        self.dates = pd.bdate_range("2024-01-02", periods=9)
        index = pd.MultiIndex.from_product(
            [self.dates, ["A", "B", "C"]], names=["date", "ticker"]
        )
        values = np.exp(np.arange(9)[:, None] * np.array([0.01, 0.02, 0.03]))
        self.prices = pd.DataFrame({"adj_close": values.ravel()}, index=index)

    def test_hand_calculated_alignment_and_warmup(self):
        samples, daily = calculate(
            self.prices, window=2, horizon=2, gap=1, min_tickers=3
        )
        expected_dates = self.dates[2:6]
        self.assertEqual(
            list(samples.index.get_level_values("date").unique()), list(expected_dates)
        )
        row = samples.loc[(self.dates[2], "B")]
        self.assertAlmostEqual(row["momentum"], 0.04)
        self.assertAlmostEqual(row["label"], np.exp(0.04) - 1)
        np.testing.assert_allclose(daily.loc[expected_dates, "rank_ic"], 1)
        self.assertTrue((daily.loc[expected_dates, "n_tickers"] == 3).all())

    def test_future_price_changes_only_label_at_signal_date(self):
        original, _ = calculate(self.prices, window=2, horizon=2, min_tickers=3)
        changed = self.prices.copy()
        changed.loc[(self.dates[5], "B"), "adj_close"] *= 2
        updated, _ = calculate(changed, window=2, horizon=2, min_tickers=3)
        key = (self.dates[2], "B")
        self.assertEqual(original.loc[key, "momentum"], updated.loc[key, "momentum"])
        self.assertNotEqual(original.loc[key, "label"], updated.loc[key, "label"])

    def test_missing_sample_falls_below_cross_section_threshold(self):
        self.prices.loc[(self.dates[5], "C"), "adj_close"] = np.nan
        samples, daily = calculate(
            self.prices, window=2, horizon=2, gap=1, min_tickers=3
        )
        self.assertNotIn((self.dates[2], "C"), samples.index)
        self.assertEqual(daily.loc[self.dates[2], "n_tickers"], 2)
        self.assertTrue(pd.isna(daily.loc[self.dates[2], "rank_ic"]))

    def test_trace_writes_results_and_call_chain(self):
        calculate(self.prices, window=2, horizon=2, min_tickers=3)
        with redirect_stdout(io.StringIO()):
            print_call_chain()
        text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("calculate 耗时", text)
        self.assertIn("momentum", text)
        self.assertIn("label", text)
        self.assertIn("调用链:", text)

    def test_trace_records_and_reraises_failure(self):
        missing = Path(self.log_directory.name) / "missing.db"
        with patch("experiment.DB_PATH", missing), self.assertRaises(FileNotFoundError):
            main()
        text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("main 耗时", text)
        self.assertIn("FileNotFoundError", text)


if __name__ == "__main__":
    unittest.main()
