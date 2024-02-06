#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

# 3-8 decoder 

from utils import *

NAME = Path(__file__).stem

dataset = [
  ('000', [0, 0, 0, 0, 0, 0, 0, 1]),
  ('001', [0, 0, 0, 0, 0, 0, 1, 0]),
  ('010', [0, 0, 0, 0, 0, 1, 0, 0]),
  ('011', [0, 0, 0, 0, 1, 0, 0, 0]),
  ('100', [0, 0, 0, 1, 0, 0, 0, 0]),
  ('101', [0, 0, 1, 0, 0, 0, 0, 0]),
  ('110', [0, 1, 0, 0, 0, 0, 0, 0]),
  ('111', [1, 0, 0, 0, 0, 0, 0, 0]),
]


# Basic Encoder + Hardware-Efficient Ansatz
def get_circuit(qv:List[Qubit], cv:List[CBit], X:List[str], params:List[float]) -> QProg:
  prog = QProg() \
    << get_basis_encoder(qv, X) \
    << get_HAE(qv, params) \
    << measure_all(qv, cv)
  return prog


# train on simulator
params = train(dataset, get_circuit, maxiter=100, shot=3000)
# infer on real chip
infer(dataset, get_circuit, params, NAME, shot=1000, use_real=True)
