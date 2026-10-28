"""用手算行情检查复权恒等式, 标签位置, 补齐日和选择规则"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from experiment import (
    choose_factor,
    daily_rank_ic,
    demean_by_industry,
    forward_quintiles,
    make_factors,
    make_label,
    paired_bootstrap,
    period_dates,
    sample_mask,
    second_look_choice,
)
from followup import followup_record, require_same_fingerprint, selection_table, stamp_fingerprint
from quanttoolkit.panel.returns import log_returns
from quanttoolkit.tracelog import configure_log, print_call_chain

WINDOWS = {
    "mom_window": 3,
    "skip": 1,
    "long_window": 4,
    "pressure_window": 2,
    "volume_window": 2,
}


def panel(periods=16, tickers=("A", "B", "C")):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    frames = []
    for offset, ticker in enumerate(tickers):
        step = 0.01 * (offset + 1)
        adj = np.exp(np.arange(periods) * step)
        close = adj.copy()
        close[: periods // 2] *= 2
        opening = close * np.exp(0.001 * (np.arange(periods) - offset))
        high = np.maximum(close, opening) * 1.01
        low = np.minimum(close, opening) * 0.99
        volume = 1000.0 + offset * 10 + np.arange(periods) * 5.0
        index = pd.MultiIndex.from_product(
            [dates, [ticker]], names=["date", "ticker"]
        )
        frames.append(
            pd.DataFrame(
                {
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": adj,
                    "volume": volume,
                },
                index=index,
            )
        )
    return pd.concat(frames).sort_index()


class FactorSearchTests(unittest.TestCase):
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

    def test_intraday_plus_overnight_matches_momentum_after_split(self):
        prices = panel()
        factors = make_factors(prices, **WINDOWS)
        reconstructed = (
            factors["intraday_strength_63"] - factors["overnight_reversal_63"]
        )
        both = reconstructed.notna() & factors["mom_63"].notna()
        self.assertTrue(both.any())
        np.testing.assert_allclose(
            reconstructed[both], factors["mom_63"][both], atol=1e-12
        )
        dates = prices.index.get_level_values("date").unique()
        history = prices.xs("A", level="ticker").loc[: dates[5]].tail(3)
        previous = prices.xs("A", level="ticker")["adj_close"].shift(1).loc[history.index]
        adjusted_open = history["open"] * history["adj_close"] / history["close"]
        expected = -np.log(adjusted_open / previous).sum()
        self.assertAlmostEqual(
            factors["overnight_reversal_63"].loc[(dates[5], "A")], expected
        )

    def test_low_volatility_is_negative_log_return_std(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
        adj = np.array([1.0, np.e, np.e, np.e**2])
        prices = pd.DataFrame(
            {
                "open": adj,
                "high": adj * 1.01,
                "low": adj * 0.99,
                "close": adj,
                "adj_close": adj,
                "volume": np.ones(4),
            },
            index=index,
        )
        factor = make_factors(
            prices,
            mom_window=2,
            skip=0,
            long_window=1,
            pressure_window=2,
            volume_window=2,
        )["low_volatility_63"]
        self.assertAlmostEqual(factor.iloc[-1], -np.sqrt(0.5))
        manual = -_rolling_std(log_returns(df=prices), 2)
        np.testing.assert_allclose(factor, manual, equal_nan=True)

    def test_label_uses_next_record_and_filled_day_drops_signal(self):
        prices = panel()
        dates = prices.index.get_level_values("date").unique()
        label = make_label(prices, 2)
        entry = prices.loc[(dates[5], "A"), "adj_close"]
        exit_price = prices.loc[(dates[7], "A"), "adj_close"]
        self.assertAlmostEqual(label.loc[(dates[4], "A")], exit_price / entry - 1)
        factors = make_factors(prices, **WINDOWS)
        mask = sample_mask(
            prices, factors, label, live_bars=4, history_bars=5, horizon=2
        )
        self.assertTrue(mask.loc[(dates[6], "B")])
        changed = prices.copy()
        changed.loc[(dates[8], "A"), "volume"] = 0
        changed_factors = make_factors(changed, **WINDOWS)
        changed_mask = sample_mask(
            changed,
            changed_factors,
            make_label(changed, 2),
            live_bars=4,
            history_bars=5,
            horizon=2,
        )
        self.assertFalse(changed_mask.loc[(dates[8], "A")])
        self.assertFalse(changed_mask.loc[(dates[7], "A")])

    def test_price_volume_treats_zero_volume_outside_live_window_as_missing(self):
        prices = panel()
        dates = prices.index.get_level_values("date").unique()
        prices.loc[(dates[2], "A"), "volume"] = 0
        factors = make_factors(
            prices,
            mom_window=3,
            skip=1,
            long_window=4,
            pressure_window=2,
            volume_window=3,
        )
        self.assertTrue(np.isnan(factors["price_volume_divergence_63"].loc[(dates[6], "A")]))
        self.assertTrue(np.isfinite(factors["mom_231_skip_21"].loc[(dates[6], "A")]))

    def test_factors_do_not_read_prices_after_the_signal_date(self):
        prices = panel()
        dates = prices.index.get_level_values("date").unique()
        before = make_factors(prices, **WINDOWS)
        changed = prices.copy()
        later = changed.index.get_level_values("date") >= dates[10]
        for column in changed.columns:
            changed.loc[later, column] *= 7
        after = make_factors(changed, **WINDOWS)
        kept = before.index.get_level_values("date") < dates[10]
        for name in before:
            pd.testing.assert_series_equal(before[name].loc[kept], after[name].loc[kept])
        self.assertAlmostEqual(
            before["mom_63"].loc[(dates[8], "A")], after["mom_63"].loc[(dates[8], "A")]
        )
        self.assertNotAlmostEqual(
            make_label(prices, 2).loc[(dates[8], "A")],
            make_label(changed, 2).loc[(dates[8], "A")],
        )

    def test_period_purges_labels_crossing_boundary(self):
        calendar = pd.bdate_range("2022-11-01", "2023-02-01")
        selected = period_dates(calendar, "selection", horizon=21)
        exits = pd.Series(calendar, index=calendar).shift(-22)
        self.assertTrue(len(selected) > 0)
        self.assertTrue((exits.loc[selected] <= pd.Timestamp("2022-12-31")).all())
        self.assertFalse((selected >= "2023-01-01").any())

    def test_rank_ic_ties_and_cross_section_floor(self):
        date = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product(
            [[date], [f"T{i}" for i in range(120)]], names=["date", "ticker"]
        )
        factor = pd.Series(np.tile([1.0, 1.0, 3.0], 40), index=index)
        label = pd.Series(np.tile([3.0, 2.0, 1.0], 40), index=index)
        mask = pd.Series(True, index=index)
        score = daily_rank_ic(factor, label, mask, min_names=100)
        self.assertTrue(np.isclose(score.iloc[0], -np.sqrt(0.75)))
        mask.iloc[:30] = False
        self.assertTrue(np.isnan(daily_rank_ic(factor, label, mask, min_names=100).iloc[0]))

    def test_quintiles_and_industry_mean(self):
        date = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product(
            [[date], list("ABCDE")], names=["date", "ticker"]
        )
        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
        label = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=index)
        daily = forward_quintiles(
            factor, label, pd.Series(True, index=index), min_names=5
        )
        self.assertAlmostEqual(daily.loc[date, 5] - daily.loc[date, 1], 0.4)
        industry = pd.Series(["Tech", "Tech", "Bank"], index=list("ABC"))
        neutral = demean_by_industry(
            pd.Series([1.0, 3.0, 10.0], index=index[:3]), industry
        )
        self.assertEqual(neutral.tolist(), [-1.0, 1.0, 0.0])

    def test_missing_industry_does_not_keep_a_two_name_correlation(self):
        date = pd.Timestamp("2024-01-02")
        tickers = [f"T{i}" for i in range(120)]
        index = pd.MultiIndex.from_product([[date], tickers], names=["date", "ticker"])
        raw = pd.Series(np.arange(120, dtype=float), index=index)
        industry = pd.Series("Tech", index=tickers[:2])
        factor = demean_by_industry(raw, industry)
        label = demean_by_industry(raw, industry)
        mask = pd.Series(True, index=index)
        score = daily_rank_ic(factor, label, mask, min_names=100)
        self.assertEqual(int(mask.sum()), 120)
        self.assertEqual(int((factor.notna() & label.notna()).sum()), 2)
        self.assertTrue(np.isnan(score.iloc[0]))
        self.assertAlmostEqual(
            daily_rank_ic(factor, label, mask, min_names=2).iloc[0], 1.0
        )

    def test_quintiles_drop_dates_below_the_same_pair_floor(self):
        date = pd.Timestamp("2024-01-02")
        tickers = [f"T{i}" for i in range(120)]
        index = pd.MultiIndex.from_product([[date], tickers], names=["date", "ticker"])
        factor = pd.Series(np.nan, index=index)
        label = pd.Series(np.nan, index=index)
        factor.iloc[:2] = [1.0, 2.0]
        label.iloc[:2] = [0.1, 0.2]
        mask = pd.Series(True, index=index)
        blocked = forward_quintiles(factor, label, mask, min_names=100)
        self.assertNotIn(date, blocked.index)
        kept = forward_quintiles(factor, label, mask, min_names=2)
        self.assertIn(date, kept.index)

    def test_followup_rejects_stale_selection_fingerprint(self):
        current = {
            "database_sha256": "new-db",
            "protocol_sha256": "proto",
            "dependency_commit": "abc",
        }
        frozen = {
            "selected": "low_volatility_63",
            "database_sha256": "old-db",
            "protocol_sha256": "proto",
            "dependency_commit": "abc",
        }
        with self.assertRaises(RuntimeError):
            require_same_fingerprint(frozen, current)
        frozen["database_sha256"] = "new-db"
        frozen["protocol_sha256"] = "old-proto"
        with self.assertRaises(RuntimeError):
            require_same_fingerprint(frozen, current)
        frozen["protocol_sha256"] = "proto"
        require_same_fingerprint(frozen, current)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selection.csv").write_text(
                "factor,period,rank_ic,days\n"
                "overnight_reversal_63,discovery,0.02,10\n"
                "overnight_reversal_63,selection,0.01,10\n",
                encoding="utf-8",
            )
            (root / "frozen_selection.json").write_text(
                json.dumps(frozen), encoding="utf-8"
            )
            stale = dict(current)
            stale["database_sha256"] = "other-db"
            with (
                patch("experiment.RESULTS", root),
                patch("experiment.fingerprints", return_value=stale),
                self.assertRaises(RuntimeError),
            ):
                selection_table(None, None)
            with (
                patch("experiment.RESULTS", root),
                patch("experiment.fingerprints", return_value=current),
            ):
                table = selection_table(None, None)
                stamped = stamp_fingerprint({})
            self.assertEqual(table.iloc[0]["factor"], "overnight_reversal_63")
            self.assertEqual(stamped["database_sha256"], "new-db")
            self.assertEqual(stamped["protocol_sha256"], "proto")
            self.assertEqual(stamped["dependency_commit"], "abc")

    def test_selection_rules_ignore_later_periods_and_benchmarks(self):
        table = pd.DataFrame(
            [
                {"factor": "mom_63", "period": "selection", "rank_ic": 0.9},
                {"factor": "low_volatility_63", "period": "selection", "rank_ic": 0.2},
                {"factor": "overnight_reversal_63", "period": "selection", "rank_ic": 0.1},
                {"factor": "low_volatility_63", "period": "discovery", "rank_ic": -0.4},
                {"factor": "overnight_reversal_63", "period": "discovery", "rank_ic": 0.05},
                {
                    "factor": "price_volume_divergence_63",
                    "period": "discovery",
                    "rank_ic": 0.02,
                },
                {
                    "factor": "price_volume_divergence_63",
                    "period": "selection",
                    "rank_ic": 0.03,
                },
            ]
        )
        self.assertEqual(choose_factor(table), "low_volatility_63")
        self.assertEqual(second_look_choice(table), "overnight_reversal_63")
        record = followup_record(table)
        self.assertTrue(record["rule_matches_overnight"])
        self.assertFalse(record["windows_or_signs_changed"])

    def test_bootstrap_is_deterministic(self):
        flat = paired_bootstrap(pd.Series([0.2, 0.2, 0.2]), block=2, repetitions=10, seed=1)
        self.assertAlmostEqual(flat[0], 0.2)
        self.assertAlmostEqual(flat[1], 0.2)
        diff = pd.Series([0.1, -0.2, 0.3, 0.0, 0.2])
        self.assertEqual(
            paired_bootstrap(diff, block=2, repetitions=40, seed=18),
            paired_bootstrap(diff, block=2, repetitions=40, seed=18),
        )

    def test_trace_writes_factor_names_and_reraises_missing_database(self):
        make_factors(panel(periods=8), **WINDOWS)
        with redirect_stdout(io.StringIO()):
            print_call_chain()
        text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("make_factors 耗时", text)
        self.assertIn("overnight_reversal_63", text)
        self.assertIn("调用链:", text)
        missing = Path(self.log_directory.name) / "missing.db"
        from experiment import main

        with patch("experiment.DB_PATH", missing), self.assertRaises(FileNotFoundError):
            main()
        self.assertIn("FileNotFoundError", self.log_path.read_text(encoding="utf-8"))


def _rolling_std(series, window):
    values = (
        series.groupby(level="ticker", sort=False)
        .rolling(window, min_periods=window)
        .std(ddof=1)
    )
    return values.droplevel(0).sort_index()


if __name__ == "__main__":
    unittest.main()
