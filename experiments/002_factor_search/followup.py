"""首轮冻结之后, 按任务指定再查看隔夜反转

这一步使用发现期和选择期才能解释的追加规则
检查期和留出期的结果不能当作独立确认
"""

import json
from pathlib import Path

import pandas as pd

import experiment
from quanttoolkit.tracelog import configure_log, trace

OUT = Path(__file__).resolve().parent / "followup_overnight"
LOG_PATH = Path(__file__).resolve().parent / "trace.log"


def followup_record(table):
    """记下追加规则在发现期和选择期会选中谁"""
    rule_choice = experiment.second_look_choice(table)
    return {
        "factor": "overnight_reversal_63",
        "formula": (
            "-sum(log((open[t] * adj_close[t] / close[t]) / adj_close[t-1]), 63)"
        ),
        "selection_reason": (
            "任务指定的追加候选"
            ", 原规则是发现期和选择期都为正的候选里较小 Rank IC 最高"
        ),
        "rule_choice_from_discovery_and_selection": rule_choice,
        "rule_matches_overnight": rule_choice == "overnight_reversal_63",
        "timing": "写在首轮冻结之后, 不回头更换首轮因子",
        "status": "探索性的第二次查看, 不是未被触碰的样本外确认",
        "windows_or_signs_changed": False,
        "comparison_mask": "八个因子的共同掩码, 与首轮相同",
        "no_further_candidate_switches": True,
    }


def selection_table(prices, factors):
    """优先读首轮 selection.csv, 避免另算一套发现期和选择期"""
    saved = experiment.RESULTS / "selection.csv"
    if saved.exists():
        return pd.read_csv(saved)
    rows = []
    label = experiment.make_label(prices, experiment.HORIZON)
    mask = experiment.sample_mask(prices, factors, label)
    for name, frame in factors.items():
        score = experiment.daily_rank_ic(frame, label, mask)
        for period in ("discovery", "selection"):
            part = experiment._slice(
                score,
                experiment.period_dates(score.index, period, experiment.HORIZON),
            )
            rows.append(
                {
                    "factor": name,
                    "period": period,
                    "rank_ic": part.mean(),
                    "days": len(part),
                }
            )
    return pd.DataFrame(rows)


@trace
def main():
    out = OUT
    out.mkdir(exist_ok=True)
    prices, industry = experiment.load_prices()
    factors = experiment.make_factors(prices)
    record = followup_record(selection_table(prices, factors))
    (out / "followup_protocol.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    comparison, uncertainty = experiment.evaluate(
        prices, factors, industry, "overnight_reversal_63", out
    )
    print(comparison.to_string(index=False), flush=True)
    print(uncertainty.to_string(index=False), flush=True)


if __name__ == "__main__":
    configure_log(LOG_PATH)
    main()
