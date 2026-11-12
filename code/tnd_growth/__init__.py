"""tnd_growth：TND 固定随机拓扑 vs 按利用率生长的最小实现。

模块：
  model.py      TND 网络（固定随机拓扑；前向与上游一致）
  data.py       Pong 行为克隆数据切批 + seed 驱动的 batch 序
  probe.py      利用率探针 A/G/S 与剪/长候选规则
  rewire.py     重连算子（剪、长 k 换 k、谱归一化）
  pong.py       纯 Python Pong 评测环境（无 pygame 依赖）
  experiment.py 四臂执行器（训练→探针→重连→再训练→评测→时延）
"""
__version__ = "1.0.0"

from .model import TND  # noqa: F401
from .probe import ProbeTND, probe  # noqa: F401