#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

# 3-qubit QFT
# https://en.wikipedia.org/wiki/Quantum_Fourier_transform

from utils import *

NAME = Path(__file__).stem

USE_REAL = True

if USE_REAL:
  qvm, qv, cv = get_qvm(3)
else:
  qvm, qv, cv = get_qvm_sim(3)

prog = QProg() \
  << H(qv[0]) \
  << CR(qv[1], qv[0], 2*pi/2**2) \
  << CR(qv[2], qv[0], 2*pi/2**3) \
  << H(qv[1]) \
  << CR(qv[2], qv[1], 2*pi/2**2) \
  << H(qv[2]) \
  << measure_all(qv, cv)

run_prog(NAME, qvm, prog) if USE_REAL else run_prog_sim(qvm, prog)
