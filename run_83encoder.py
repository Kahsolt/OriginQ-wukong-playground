#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

# 8-3 encoder 

from utils import *

NAME = Path(__file__).stem

dataset = [
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0],
]
dataset = [np.asarray(x, dtype=np.float32).tolist() for x in dataset]
dataset = [(x, x) for x in dataset]


# Amplitude Encoder + Hardware-Efficient Ansatz
def get_circuit(qv:List[Qubit], cv:List[CBit], X:List[float], params:List[float]) -> QProg:
  prog = QProg() \
    << get_amplitude_encoder(qv, X) \
    << get_HAE(qv, params) \
    << measure_all(qv, cv)
  return prog


# train on simulator
params = train(dataset, get_circuit, maxiter=100, shot=3000)
# infer on real chip
infer(dataset, get_circuit, params, NAME, shot=1000, use_real=True)
