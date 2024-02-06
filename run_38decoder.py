#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/06

# 3-8 decoder 

from scipy.optimize import minimize, Bounds
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
def get_circuit(qv:List[Qubit], cv:List[CBit], X:List[float], params:List[float]) -> QProg:
  encoder = Encode()
  encoder.basic_encode(qv, X)
  prog = QProg() \
    << encoder.get_circuit() \
    << RX(qv[0], params[0]) \
    << RY(qv[0], params[1]) \
    << RX(qv[1], params[2]) \
    << RY(qv[1], params[3]) \
    << RX(qv[2], params[4]) \
    << RY(qv[2], params[5]) \
    << CNOT(qv[0], qv[1]) \
    << CNOT(qv[1], qv[2]) \
    << RX(qv[0], params[6]) \
    << RY(qv[0], params[7]) \
    << RX(qv[1], params[8]) \
    << RY(qv[1], params[9]) \
    << RX(qv[2], params[10]) \
    << RY(qv[2], params[11]) \
    << CNOT(qv[0], qv[1]) \
    << CNOT(qv[1], qv[2]) \
    << RX(qv[0], params[12]) \
    << RY(qv[0], params[13]) \
    << RX(qv[1], params[14]) \
    << RY(qv[1], params[15]) \
    << RX(qv[2], params[16]) \
    << RY(qv[2], params[17]) \
    << measure_all(qv, cv)
  return prog


def train(maxiter=100, shot:int=3000) -> List[float]:
  def func(params:np.ndarray) -> float:
    nonlocal n_iter
    n_iter += 1
    if n_iter > maxiter: raise ValueError(f'exceed MAX_ITER: {n_iter} > {maxiter}')

    losses = []
    for X, Y in dataset:
      qvm, qv, cv = get_qvm_sim(3)
      prog = get_circuit(qv, cv, X, params)
      res = run_prog_sim(qvm, prog, shot=shot, log=False)
      probs = np.asarray([res.get(k, 0) for k in [bin(i)[2:].rjust(3, '0') for i in range(len(dataset))]])
      loss = ((probs - Y) ** 2).mean()
      losses.append(loss)
    loss_avg = sum(losses) / len(losses)
    if n_iter % 10 == 0:
      print(f'>> [{n_iter}/{maxiter}] loss: {loss_avg}')
    return loss_avg

  n_iter = 0
  params = np.random.uniform(low=-pi/32, high=pi/32, size=[18])
  res = minimize(func, params, method='COBYLA', tol=1e-8, bounds=Bounds(-pi, pi), options={'maxiter': maxiter, 'disp': True})
  print('best f(x):', res.fun)
  print('best x:', res.x)
  return res.x


def infer(params:List[float], shot:int=1000, use_real:bool=False):
  log_fp = LOG_PATH / (NAME + '.json')
  db = load_db(log_fp)
  try:
    if use_real:
      qvm, qv, cv = get_qvm(3)
    else:
      qvm, qv, cv = get_qvm_sim(3)
    for X, Y in dataset:
      prog = get_circuit(qv, cv, X, params)
      if use_real:
        res = qvm.real_chip_measure(prog, shot, real_chip_type.origin_72, is_amend=False, is_mapping=False, is_optimization=False)
      else:
        res = run_prog_sim(qvm, prog, shot=shot)
      probs = np.asarray([res.get(k, 0) for k in [bin(i)[2:].rjust(3, '0') for i in range(len(dataset))]])
      print(probs)
      db.append({
        'X': X,
        'res': dict(res),
        'probs': list(probs),
      })
      save_db(log_fp, db)
  except:
    print_exc()
  finally:
    save_db(log_fp, db)


# train on simulator
params = train()
# infer on real chip
infer(params, use_real=True)
