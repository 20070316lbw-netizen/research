"""按预先写好的协议比较六个量价候选, 只按选择期冻结一个"""

import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import pandas as pd
from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.factors.neutral import neutralize_by_industry
from quanttoolkit.metrics import ic_summary, rank_ic
from quanttoolkit.panel.returns import future_returns, log_returns
from quanttoolkit.panel.transfer import to_date_ticker_frame
from quanttoolkit.tracelog import configure_log, trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
LOG_PATH = HERE / "trace.log"
DB_PATH = ROOT / "data" / "quanttoolkit.db"
PROTOCOL_PATH = HERE / "protocol.json"

BENCHMARKS = ["mom_63", "mom_231_skip_21"]
CANDIDATES = [
    "intraday_strength_63",
    "overnight_reversal_63",
    "closing_pressure_21",
    "price_volume_divergence_63",
    "low_volatility_63",
    "smooth_momentum_63",
]
PERIODS = {
    "discovery": ("2017-01-01", "2020-12-31"),
    "selection": ("2021-01-01", "2022-12-31"),
    "check": ("2023-01-01", "2024-12-31"),
    "holdout": ("2025-01-01", "2099-12-31"),
}
MOM_WINDOW = 63
SKIP = 21
LONG_WINDOW = 231
PRESSURE_WINDOW = 21
VOLUME_WINDOW = 21
LIVE_BARS = 64
HISTORY_BARS = 252
HORIZON = 21
MIN_NAMES = 100
BOOTSTRAP_BLOCK = 63
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 18
CHECK_HORIZONS = (5, 21, 63)


def _rolling(series, window, method):
    """按证券交易记录做滚动, 窗口长度不足时为 NaN"""
    values = getattr(
        series.groupby(level="ticker", sort=False).rolling(window, min_periods=window),
        method,
    )()
    return values.droplevel(0).sort_index()


def _rolling_corr(left, right, window):
    """两列按同一证券滚动相关, 窗口内有缺失则为 NaN"""
    paired = pd.concat(
        [left.rename("left"), right.reindex(left.index).rename("right")], axis=1
    )
    pieces = []
    for _, group in paired.groupby(level="ticker", sort=False):
        pieces.append(
            group["left"].rolling(window, min_periods=window).corr(group["right"])
        )
    if not pieces:
        return left.copy()
    return pd.concat(pieces).sort_index()


def _finite(series):
    return series.replace([np.inf, -np.inf], np.nan)


def period_dates(calendar, period, horizon):
    """留下标签退出日仍落在区间终点之前的信号日"""
    start, end = (pd.Timestamp(value) for value in PERIODS[period])
    days = pd.DatetimeIndex(pd.DatetimeIndex(calendar).unique()).sort_values()
    exits = pd.Series(days, index=days).shift(-(horizon + 1))
    keep = (days >= start) & (days <= end) & (exits.to_numpy() <= np.datetime64(end))
    return days[np.asarray(keep, dtype=bool)]


def make_label(prices, horizon):
    """主标签及稳健性标签, 入场在下一笔记录"""
    return future_returns(df=prices, n_periods=horizon, gap=1)


def demean_by_industry(series, industry):
    """用工具箱按子行业去掉当天均值"""
    return neutralize_by_industry(series.to_frame("value"), industry)["value"]


@trace
def make_factors(
    prices,
    *,
    mom_window=MOM_WINDOW,
    skip=SKIP,
    long_window=LONG_WINDOW,
    pressure_window=PRESSURE_WINDOW,
    volume_window=VOLUME_WINDOW,
):
    """计算两个动量基准和六个候选

    动量与低波动调用工具箱
    低波动取负号, 其余五个候选工具箱没有对应函数
    成交量为 0 的补齐日不计入隔夜, 日内, 收盘位置, 量价和路径平滑
    动量仍使用补齐后的价格, 与收盘价比值口径一致
    """
    adj = prices["adj_close"]
    live = prices["volume"] > 0
    previous_live = (
        live.groupby(level="ticker", sort=False).shift(1).fillna(False).astype(bool)
    )
    tradable = live & previous_live
    returns = log_returns(df=prices, price_col="adj_close").where(tradable)
    intraday = np.log(prices["close"] / prices["open"]).where(live)
    adjusted_open = prices["open"] * adj / prices["close"]
    previous_adj = adj.groupby(level="ticker", sort=False).shift(1)
    overnight = np.log(adjusted_open / previous_adj).where(tradable)
    spread = prices["high"] - prices["low"]
    closing = (2 * prices["close"] - prices["high"] - prices["low"]) / spread
    closing = closing.where((prices["high"] > prices["low"]) & live)
    traded_volume = prices["volume"].where(live)
    relative_volume = np.log(
        traded_volume / _rolling(traded_volume, volume_window, "mean")
    )
    smooth_num = _rolling(returns, mom_window, "sum")
    smooth_den = _rolling(returns.abs(), mom_window, "sum")
    factors = pd.concat(
        [
            base_momfactor(prices, window=mom_window, skip=0).rename("mom_63"),
            base_momfactor(prices, window=long_window, skip=skip).rename(
                "mom_231_skip_21"
            ),
            _rolling(intraday, mom_window, "sum").rename("intraday_strength_63"),
            (-_rolling(overnight, mom_window, "sum")).rename("overnight_reversal_63"),
            _rolling(closing, pressure_window, "mean").rename("closing_pressure_21"),
            (-_rolling_corr(returns, relative_volume, mom_window)).rename(
                "price_volume_divergence_63"
            ),
            (-base_logvolfactor(prices, window=mom_window)).rename("low_volatility_63"),
            (smooth_num / smooth_den).rename("smooth_momentum_63"),
        ],
        axis=1,
    )
    return _finite(factors).reindex(prices.index)


def sample_mask(
    prices,
    factors,
    label,
    *,
    live_bars=LIVE_BARS,
    history_bars=HISTORY_BARS,
    horizon=HORIZON,
):
    """八个因子和标签都有效, 且近期成交与历史长度满足协议"""
    live = prices["volume"] > 0
    grouped = live.groupby(level="ticker", sort=False)
    history = (
        prices["adj_close"]
        .groupby(level="ticker", sort=False)
        .shift(history_bars)
        .notna()
    )
    mask = (
        _rolling(live.astype(float), live_bars, "sum").eq(live_bars)
        & grouped.shift(-1).eq(True)
        & grouped.shift(-(horizon + 1)).eq(True)
        & history
        & label.notna()
    )
    for name in factors.columns:
        mask = mask & factors[name].notna()
    return mask.reindex(prices.index).fillna(False)


def pair_mask(factor, other, mask):
    """掩码内因子和比较值都非空的样本"""
    index = mask.index
    return (
        mask.reindex(index).fillna(False)
        & factor.reindex(index).notna()
        & other.reindex(index).notna()
    )


def daily_rank_ic(factor, other, mask, min_names=MIN_NAMES):
    """调用工具箱 Rank IC, 当天有效配对不足时记为缺失"""
    paired = pair_mask(factor, other, mask)
    score = rank_ic(factor=factor.where(paired), forward_return=other.where(paired))
    counts = paired.groupby(level="date").sum()
    score = score.reindex(counts.index)
    score = score.where(counts >= min_names)
    score.name = "rank_ic"
    return score


def forward_quintiles(factor, label, mask, min_names=MIN_NAMES):
    """按当天平均秩把未来收益分成五档, 返回每日各档均值

    只保留有效配对不少于 min_names 的日期, 与 Rank IC 使用同一门槛
    """
    paired = pair_mask(factor, label, mask)
    counts = paired.groupby(level="date").sum()
    eligible = counts[counts >= min_names].index
    kept = paired & paired.index.get_level_values("date").isin(eligible)
    data = pd.DataFrame({"factor": factor, "label": label}).loc[kept]
    columns = [1, 2, 3, 4, 5]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rank = data.groupby(level="date")["factor"].rank(pct=True, method="average")
    data = data.assign(quintile=np.ceil(rank * 5).clip(1, 5).astype(int))
    means = data.groupby([pd.Grouper(level="date"), "quintile"])["label"].mean()
    return means.unstack("quintile").reindex(columns=columns)


def choose_factor(table):
    """选择期平均 Rank IC 最高的候选, 基准不参与"""
    shortlist = table.loc[
        (table["period"] == "selection") & table["factor"].isin(CANDIDATES)
    ]
    ordered = shortlist.sort_values("rank_ic", ascending=False, kind="mergesort")
    return ordered.iloc[0]["factor"]


def second_look_choice(table):
    """两段都为正的候选里, 取较小一段 Rank IC 最高者

    这是首轮失败后追加的规则, 只使用发现期和选择期
    """
    wide = table.pivot(index="factor", columns="period", values="rank_ic")
    wide = wide.loc[wide.index.isin(CANDIDATES), ["discovery", "selection"]]
    positive = wide[(wide["discovery"] > 0) & (wide["selection"] > 0)]
    if positive.empty:
        return None
    score = positive.min(axis=1).sort_values(ascending=False, kind="mergesort")
    return score.index[0]


def paired_bootstrap(
    diff,
    block=BOOTSTRAP_BLOCK,
    repetitions=BOOTSTRAP_REPEATS,
    seed=BOOTSTRAP_SEED,
):
    """对日度 Rank IC 差值做移动区块自助法, 返回 95% 分位"""
    values = diff.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    count = (len(values) + block - 1) // block
    starts = rng.integers(0, len(values), size=(repetitions, count))
    indices = (starts[..., None] + np.arange(block)) % len(values)
    means = values[indices.reshape(repetitions, -1)[:, : len(values)]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _slice(score, dates):
    return score.loc[score.index.isin(dates)].dropna()


@trace
def select(prices, factors, out, *, horizon=HORIZON, min_names=MIN_NAMES):
    """只汇总发现期和选择期, 写入冻结记录后再返回"""
    label = make_label(prices, horizon)
    mask = sample_mask(prices, factors, label, horizon=horizon)
    rows = []
    for name, frame in factors.items():
        score = daily_rank_ic(frame, label, mask, min_names)
        for period in ("discovery", "selection"):
            part = _slice(score, period_dates(score.index, period, horizon))
            rows.append(
                {
                    "factor": name,
                    "period": period,
                    "rank_ic": part.mean(),
                    "days": len(part),
                }
            )
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "selection.csv", index=False)
    chosen = choose_factor(table)
    record = {
        "selected": chosen,
        "metrics_seen": ["discovery", "selection"],
    }
    record.update(fingerprints())
    (out / "frozen_selection.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return chosen, table


@trace
def evaluate(
    prices,
    factors,
    industry,
    chosen,
    out,
    *,
    horizon=HORIZON,
    min_names=MIN_NAMES,
):
    """评估已冻结因子和两个基准, 不改窗口和方向"""
    names = BENCHMARKS + [chosen]
    label = make_label(prices, horizon)
    mask = sample_mask(prices, factors, label, horizon=horizon)
    scores = pd.DataFrame(
        {name: daily_rank_ic(factors[name], label, mask, min_names) for name in names}
    )
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "daily_rank_ic.csv", float_format="%.10f")
    rows, differences, yearly, quintiles = [], [], [], []
    for period in PERIODS:
        dates = period_dates(scores.index, period, horizon)
        for name in names:
            part = _slice(scores[name], dates)
            if part.empty:
                continue
            stats = ic_summary(ic=part)
            groups = forward_quintiles(
                factors[name], label, mask, min_names=min_names
            )
            groups = groups.loc[groups.index.isin(dates)]
            pair_counts = pair_mask(factors[name], label, mask).groupby(level="date").sum()
            rows.append(
                {
                    "factor": name,
                    "period": period,
                    "rank_ic": stats["ic_mean"],
                    "ic_ir_unannualized": stats["ic_ir"],
                    "ic_positive": stats["ic_pos"],
                    "days": len(part),
                    "first": str(part.index.min().date()),
                    "last": str(part.index.max().date()),
                    "mean_stocks": float(pair_counts.reindex(part.index).mean()),
                    "q5_minus_q1_forward_return": float(
                        (groups[5] - groups[1]).mean()
                    ),
                }
            )
            for group, value in groups.mean().items():
                quintiles.append(
                    {
                        "factor": name,
                        "period": period,
                        "quintile": int(group),
                        "mean_forward_return": value,
                    }
                )
        if period in ("check", "holdout"):
            for baseline in BENCHMARKS:
                diff = _slice(scores[chosen] - scores[baseline], dates)
                low, high = paired_bootstrap(diff)
                differences.append(
                    {
                        "period": period,
                        "baseline": baseline,
                        "mean_ic_difference": diff.mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    for year in range(2017, int(scores.index.max().year) + 1):
        for name in names:
            part = scores.loc[scores.index.year == year, name].dropna()
            if part.empty:
                continue
            yearly.append(
                {
                    "factor": name,
                    "year": year,
                    "rank_ic": part.mean(),
                    "days": len(part),
                }
            )
    comparison = pd.DataFrame(rows)
    uncertainty = pd.DataFrame(differences)
    comparison.to_csv(out / "comparison.csv", index=False)
    uncertainty.to_csv(out / "paired_uncertainty.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "yearly.csv", index=False)
    pd.DataFrame(quintiles).to_csv(out / "quintiles.csv", index=False)
    robustness = []
    for check_horizon in CHECK_HORIZONS:
        check_label = make_label(prices, check_horizon)
        check_mask = sample_mask(
            prices, factors, check_label, horizon=check_horizon
        )
        for adjusted in (False, True):
            other = (
                demean_by_industry(check_label.where(check_mask), industry)
                if adjusted
                else check_label
            )
            for name in names:
                signal = (
                    demean_by_industry(factors[name].where(check_mask), industry)
                    if adjusted
                    else factors[name]
                )
                score = daily_rank_ic(signal, other, check_mask, min_names)
                for period in ("check", "holdout"):
                    part = _slice(
                        score, period_dates(score.index, period, check_horizon)
                    )
                    robustness.append(
                        {
                            "factor": name,
                            "horizon": check_horizon,
                            "industry_demeaned": adjusted,
                            "period": period,
                            "rank_ic": part.mean(),
                            "days": len(part),
                        }
                    )
    pd.DataFrame(robustness).to_csv(out / "robustness.csv", index=False)
    correlations = []
    for baseline in BENCHMARKS:
        score = daily_rank_ic(factors[chosen], factors[baseline], mask, min_names)
        for period in ("check", "holdout"):
            part = _slice(score, period_dates(score.index, period, horizon))
            correlations.append(
                {
                    "baseline": baseline,
                    "period": period,
                    "rank_correlation": part.mean(),
                }
            )
    pd.DataFrame(correlations).to_csv(out / "factor_correlations.csv", index=False)
    return comparison, uncertainty


def _digest():
    with DB_PATH.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprints():
    """当前数据库, 协议和已安装工具箱提交"""
    info = json.loads(distribution("quanttoolkit").read_text("direct_url.json"))
    return {
        "database_sha256": _digest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "dependency_commit": info["vcs_info"]["commit_id"],
    }


def _records(frame):
    rows = []
    for raw in frame.to_dict(orient="records"):
        row = {}
        for key, value in raw.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            row[key] = value
        rows.append(row)
    return rows


def write_summary(prices, chosen, comparison, uncertainty, out):
    """保存数据库指纹, 依赖提交和本轮汇总"""
    identity = fingerprints()
    payload = {
        "database": str(DB_PATH.relative_to(ROOT)),
        "database_sha256": identity["database_sha256"],
        "dependency": json.loads(
            distribution("quanttoolkit").read_text("direct_url.json")
        ),
        "selected": chosen,
        "protocol_sha256": identity["protocol_sha256"],
        "dependency_commit": identity["dependency_commit"],
        "price_rows": len(prices),
        "tickers": int(prices.index.get_level_values("ticker").nunique()),
        "price_start": str(prices.index.get_level_values("date").min().date()),
        "price_end": str(prices.index.get_level_values("date").max().date()),
        "horizon": HORIZON,
        "gap": 1,
        "min_names": MIN_NAMES,
        "comparison": _records(comparison),
        "uncertainty": _records(uncertainty),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (out / "summary.json").write_text(text, encoding="utf-8")
    print(text)


@trace
def load_prices():
    """只读本项目行情和子行业"""
    with get_duckdb(path=DB_PATH, read_only=True) as connection:
        raw = connection.execute(
            "SELECT date, ticker, open, high, low, close, adj_close, volume "
            "FROM prices ORDER BY date, ticker"
        ).df()
        universe = connection.execute(
            "SELECT ticker, sub_industry FROM universe"
        ).df()
    prices = to_date_ticker_frame(df=raw)
    industry = universe.drop_duplicates("ticker").set_index("ticker")["sub_industry"]
    return prices, industry


@trace
def main():
    out = RESULTS
    prices, industry = load_prices()
    factors = make_factors(prices)
    chosen, table = select(prices, factors, out)
    print(table.to_string(index=False), flush=True)
    print(f"FROZEN: {chosen}", flush=True)
    comparison, uncertainty = evaluate(prices, factors, industry, chosen, out)
    write_summary(prices, chosen, comparison, uncertainty, out)
    print(comparison.to_string(index=False), flush=True)
    print(uncertainty.to_string(index=False), flush=True)


if __name__ == "__main__":
    configure_log(LOG_PATH)
    main()
