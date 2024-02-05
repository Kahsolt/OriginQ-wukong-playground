#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from utils import *

NAME = Path(__file__).stem

qvm, qv, cv = get_qvm(1)
prog = QProg() \
  << H(qv[0]) \
  << measure_all(qv, cv)

run_prog(NAME, qvm, prog)
