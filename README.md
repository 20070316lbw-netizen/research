# research

个人量化实验区, 复用相邻 `../QuantToolKit` 的工具, 工作约定见 [agent.md](agent.md)。
当前通过 Git 依赖安装 `quanttoolkit`, 版本由 `uv.lock` 固定。

第一个实验: [21 期动量与未来 5 期收益](experiments/001_momentum/README.md)。
第二个实验: [量价因子搜索](experiments/002_factor_search/README.md)。
在本目录运行, 使用已有的 `data/quanttoolkit.db`:

```bash
uv run --locked python experiments/001_momentum/experiment.py
uv run --locked python -m unittest discover -s experiments/001_momentum -p 'test_*.py'
```

数据库和日志仅保留本机, 小体积实验结果随代码保存。
