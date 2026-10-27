# 001: 21 期动量与未来 5 期收益

目的: 用已有行情跑通读取 → 因子 → 标签 → 每日 Rank IC → 汇总的第一条流程。
固定下面的口径直接运行一次, 不做参数搜索。

| 项目 | 约定 |
| --- | --- |
| 数据 | 本项目 `data/quanttoolkit.db` 的 prices 表, 只读 |
| 读取区间 | 2023-10-01 至 2026-01-31, 为预热与尾部标签留数据 |
| 评估日期 | 2024-01-01 至 2025-12-31, 两端包含 |
| 证券 | 查询区间内数据库已有的全部证券, 不另挑收益好的股票 |
| 价格 | adj_close, 不另做填补或剔除已有的补齐记录 |
| 特征 | `base_momfactor(window=21, skip=0)`, 完整窗口的对数收益之和 |
| 标签 | `future_returns(n_periods=5, gap=1)`, 即 `P[t+6]/P[t+1]-1` |
| 时序 | 以证券自身交易记录计数, t 收盘后形成特征, 标签从 t+1 收盘到 t+6 收盘 |
| 有效样本 | 因子与标签成对非空, 每天至少 30 只证券 |
| 指标 | 每日截面 Spearman Rank IC, 汇总均值, 标准差, 非年化 ICIR, 正值比例 |

在项目根目录运行:

```bash
uv run --locked python experiments/001_momentum/experiment.py
uv run --locked python -m unittest discover -s experiments/001_momentum -p 'test_*.py'
```

结果写入本实验的 `results/daily_ic.csv` 和 `results/summary.json`。
使用工具箱 `quanttoolkit.tracelog` 追踪 `main`, `load_prices` 和 `calculate`,
在终端和本实验目录的 `trace.log` 输出耗时, 返回摘要, 异常及调用链。
日志路径由脚本自身位置确定, 每次运行追加记录, 日志仅留本机, 不纳入 Git。
仅被 `@trace` 装饰的步骤出现在调用链中, 内部工具箱函数不会自动展开;
`calculate` 的元组返回值沿用工具箱现有的截断摘要。
汇总同时保存输入数据库的 SHA256, 参数和已安装依赖的 Git 提交信息。
数据库不会推送, 其他机器需自行准备同口径数据; SHA256 用于核对本次输入文件。

2026-10-27 本机运行结果: 500 只证券, 249987 对有效样本, 502 个有效评估日,
每天至少 495 只证券。平均 Rank IC 为 0.001728, 非年化 ICIR 为 0.009518,
正 IC 日期占比为 53.39%。本次结果的平均相关接近零, 不支持直接宣称有预测优势。
精确数值见 [summary.json](results/summary.json), 每日明细见 [daily_ic.csv](results/daily_ic.csv)。

这是探索性的流程占位, 没有训练, 参数选择, 独立留出验证或交易回测。
当前成员名单回溯历史可能带来生存者偏差, 已有补齐记录也会影响窗口。
每日 5 期标签存在重叠, ICIR 没有年化, 不据此宣称统计显著或策略可盈利。
