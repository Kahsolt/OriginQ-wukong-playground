#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

# iris 3-class classification

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from utils import *

NAME = Path(__file__).stem

nq = 4

X, Y = load_iris(return_X_y=True)
scaler = MinMaxScaler(feature_range=(-pi/2, pi/2))
X = scaler.fit_transform(X)
Y = [np.eye(nq**2)[y] for y in Y]
dataset = [(x.tolist(), y.tolist()) for x, y in zip(X, Y)]


# Angle Encoder + Hardware-Efficient Ansatz
def get_circuit(qv:List[Qubit], cv:List[CBit], X:List[str], params:List[float]) -> QProg:
  prog = QProg() \
    << get_angle_encoder(qv, X) \
    << get_HAE(qv, params) \
    << measure_all(qv, cv)
  return prog


# train on simulator
params = train(dataset, get_circuit, maxiter=100, shot=3000, nq=nq)

# 剩余时长不够了，大约只能跑60条 :(
import random
random.shuffle(dataset)
dataset_60 = dataset[:60]

# infer on real chip
infer_clf(dataset_60, get_circuit, params, NAME, nq=nq, shot=1000, use_real=True)
