#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

from utils import *

NAME = Path(__file__).stem

USE_REAL = True

data = '101101'
print('data:', data)

nq = len(data)
if USE_REAL:
  qvm, qv, cv = get_qvm(nq)
else:
  qvm, qv, cv = get_qvm_sim(nq)

encoder = Encode()
encoder.basic_encode(qv, data)
prog = QProg() \
  << encoder.get_circuit() \
  << measure_all(qv, cv)
run_prog(NAME, qvm, prog) if USE_REAL else run_prog_sim(qvm, prog)
