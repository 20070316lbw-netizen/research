"""用本地行情跑通动量因子与未来收益的 Rank IC 评估"""

import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

import pandas as pd
from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.metrics import ic_summary, rank_ic
from quanttoolkit.panel.returns import future_returns
from quanttoolkit.panel.transfer import to_date_ticker_frame
from quanttoolkit.tracelog import configure_log, trace

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "results"
LOG_PATH = Path(__file__).resolve().parent / "trace.log"
DB_PATH = ROOT / "data" / "quanttoolkit.db"
READ_START, READ_END = "2023-10-01", "2026-01-31"
EVAL_START, EVAL_END = "2024-01-01", "2025-12-31"
WINDOW, HORIZON, GAP, MIN_TICKERS = 21, 5, 1, 30


@trace
def calculate(
    prices, *, window=WINDOW, horizon=HORIZON, gap=GAP, min_tickers=MIN_TICKERS
):
    """先在完整历史上计算, 按日期保留成对样本数与 Rank IC

    prices 为含 adj_close 的 [date, ticker] 面板
    window, horizon, gap 按每只证券的交易记录计数
    返回 samples 与 daily, samples 仅含因子和标签都非空的样本
    daily 含 n_tickers, rank_ic, 少于 min_tickers 时相关记为 NaN
    """
    factor = base_momfactor(prices, window=window).rename("momentum")
    label = future_returns(df=prices, n_periods=horizon, gap=gap).rename("label")
    samples = pd.concat([factor, label], axis=1).dropna()
    counts = samples.groupby(level="date").size().rename("n_tickers")
    daily = pd.concat(
        [counts, rank_ic(factor=factor, forward_return=label)], axis=1, sort=True
    )
    daily.loc[daily["n_tickers"] < min_tickers, "rank_ic"] = float("nan")
    return samples, daily


@trace
def load_prices():
    """只读查询本实验的行情, 返回标准价格面板"""
    with get_duckdb(path=DB_PATH, read_only=True) as connection:
        raw = connection.execute(
            "SELECT date, ticker, adj_close FROM prices "
            "WHERE date BETWEEN ? AND ? ORDER BY date, ticker",
            [READ_START, READ_END],
        ).df()
    return to_date_ticker_frame(df=raw)


@trace
def main():
    with DB_PATH.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    prices = load_prices()
    samples, daily = calculate(prices)
    daily = daily.loc[EVAL_START:EVAL_END]
    dates = samples.index.get_level_values("date")
    evaluated = samples.loc[(dates >= EVAL_START) & (dates <= EVAL_END)]
    valid = daily["rank_ic"].dropna()
    summary = {
        "database": str(DB_PATH.relative_to(ROOT)),
        "database_sha256": digest,
        "dependency": json.loads(
            distribution("quanttoolkit").read_text("direct_url.json")
        ),
        "read_start": READ_START,
        "read_end": READ_END,
        "evaluation_start": EVAL_START,
        "evaluation_end": EVAL_END,
        "window": WINDOW,
        "horizon": HORIZON,
        "gap": GAP,
        "min_tickers": MIN_TICKERS,
        "price_rows": len(prices),
        "paired_samples": len(evaluated),
        "tickers": evaluated.index.get_level_values("ticker").nunique(),
        "valid_ic_days": len(valid),
        "first_ic_date": str(valid.index.min().date()),
        "last_ic_date": str(valid.index.max().date()),
        "min_daily_tickers": int(daily.loc[valid.index, "n_tickers"].min()),
        "metrics": ic_summary(ic=valid).to_dict(),
    }
    OUTPUT.mkdir(exist_ok=True)
    daily.to_csv(OUTPUT / "daily_ic.csv", float_format="%.10f")
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    (OUTPUT / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    configure_log(LOG_PATH)
    main()
