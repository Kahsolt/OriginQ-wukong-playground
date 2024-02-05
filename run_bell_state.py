#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from utils import *

NAME = Path(__file__).stem

qvm, qv, cv = get_qvm(2)
prog = QProg() \
  << H(qv[0]) \
  << CNOT(qv[0], qv[1]) \
  << measure_all(qv, cv)

run_prog(NAME, qvm, prog)
