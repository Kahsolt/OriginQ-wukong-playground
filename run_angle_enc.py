#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from utils import *

NAME = Path(__file__).stem

USE_REAL = True

data = [pi/3, -2*pi/3, 5*pi/7]
print('data:', data)

nq = len(data)
if USE_REAL:
  qvm, qv, cv = get_qvm(3)
else:
  qvm, qv, cv = get_qvm_sim(3)

encoder = Encode()
encoder.angle_encode(qv, data)
prog = QProg() \
  << encoder.get_circuit() \
  << measure_all(qv, cv)
run_prog(NAME, qvm, prog) if USE_REAL else run_prog_sim(qvm, prog)
