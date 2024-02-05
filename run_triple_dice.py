#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from utils import *

NAME = Path(__file__).stem

qvm, qv, cv = get_qvm(2)
prog = QProg() \
  << RX(qv[0], -0.9316580014276875) \
  << RY(qv[1], -2.407622486317919) \
  << CNOT(qv[0], qv[1]) \
  << RX(qv[0], 0.41544503161253465) \
  << RY(qv[1], 1.0060027549603225) \
  << measure_all(qv, cv)

run_prog(NAME, qvm, prog)
