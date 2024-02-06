#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from utils import *

NAME = Path(__file__).stem

USE_REAL = True

# NOTE: use the official demo, other examples DOSE NOT work, maybe bugs :(
# expect: {'001': 0.355, '101': 0.327, '110': 0.318}
data = np.asarray([0, 1/np.sqrt(3), 0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 0])
data = data.tolist()
print('data:', data)

if USE_REAL:
  qvm, qv, cv = get_qvm(3)
else:
  qvm, qv, cv = get_qvm_sim(3)

encoder = Encode()
encoder.amplitude_encode(qv, data)
prog = QProg() \
  << encoder.get_circuit() \
  << measure_all(qv, cv)
run_prog(NAME, qvm, prog) if USE_REAL else run_prog_sim(qvm, prog)
