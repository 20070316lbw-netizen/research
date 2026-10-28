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
FINGERPRINT_KEYS = ("database_sha256", "protocol_sha256", "dependency_commit")


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


def require_same_fingerprint(frozen, current):
    """首轮冻结记录必须对上当前数据库, 协议和工具箱提交"""
    missing = [key for key in FINGERPRINT_KEYS if key not in frozen]
    if missing:
        raise RuntimeError(
            "冻结记录缺少 "
            + ", ".join(missing)
            + ", 请重新运行 experiment.py 后再执行 followup.py"
        )
    mismatched = [
        key for key in FINGERPRINT_KEYS if frozen[key] != current[key]
    ]
    if mismatched:
        raise RuntimeError(
            "冻结记录与当前环境的 "
            + ", ".join(mismatched)
            + " 不一致, 请重新运行 experiment.py 后再执行 followup.py"
        )


def stamp_fingerprint(record):
    """把本次数据库, 协议和工具箱提交写进追加记录"""
    stamped = dict(record)
    stamped.update(experiment.fingerprints())
    return stamped


def selection_table(prices, factors):
    """读取首轮 selection.csv 前核对指纹, 对不上就停止"""
    saved = experiment.RESULTS / "selection.csv"
    frozen_path = experiment.RESULTS / "frozen_selection.json"
    if saved.exists():
        if not frozen_path.exists():
            raise RuntimeError(
                "存在 selection.csv 但缺少 frozen_selection.json, "
                "请重新运行 experiment.py 后再执行 followup.py"
            )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        require_same_fingerprint(frozen, experiment.fingerprints())
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
    record = stamp_fingerprint(followup_record(selection_table(prices, factors)))
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
