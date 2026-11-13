"""跨模块共享的内存数据锁。

数据集修订、样本修改与训练快照之间必须串行：训练发起时抓到的字段版本和样本，
不能在抓取过程中被另一个请求改掉，否则签认时封存下来的「输入版本」就不可信。
"""
from __future__ import annotations

import threading

STORE_LOCK = threading.RLock()
