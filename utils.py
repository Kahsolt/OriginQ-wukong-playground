#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

import sys
import json
from pathlib import Path
from traceback import print_exc
from typing import *

import numpy as np
from numpy import pi
from scipy.optimize import minimize, Bounds
from pyqpanda import CPUQVM, QCloud, real_chip_type
from pyqpanda import QCircuit, QProg, Qubit, ClassicalCondition as CBit
from pyqpanda import H, I, X, Y, Z, RX, RY, RZ, CR, CNOT, SWAP, Measure, measure_all
from pyqpanda import Encode

np.random.seed(114514)

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

API_KEY = open('API_KEY.txt').read().strip()

DB = List[Dict[str, Any]]


def load_db(fp:Path) -> DB:
  if not fp.exists(): return []
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_db(fp:Path, db:DB):
  def cvt(v:Any) -> str:
    if isinstance(v, Path): return str(v)
    return v
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False, default=cvt)


def get_qvm(nq:int, log:bool=True) -> Tuple[QCloud, List[Qubit], List[CBit]]:
  qvm = QCloud()
  qvm.set_configure(72, 72)
  qvm.init_qvm(API_KEY, is_logged=log)
  qv = qvm.qAlloc_many(nq)
  cv = qvm.cAlloc_many(nq)
  return qvm, qv, cv

def run_prog(name:str, qvm:QCloud, prog:QProg, optimize:bool=True):
  log_fp = LOG_PATH / (name + '.json')
  db = load_db(log_fp)
  try:
    res = qvm.real_chip_measure(prog, 1000, real_chip_type.origin_72, is_amend=optimize, is_mapping=optimize, is_optimization=optimize)
    print(res)
  except:
    print_exc()
  finally:
    db.append({
      'res': locals().get('res', None),
    })
    save_db(log_fp, db)


def get_qvm_sim(nq:int) -> Tuple[CPUQVM, List[Qubit], List[CBit]]:
  qvm = CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(nq)
  cv = qvm.cAlloc_many(nq)
  return qvm, qv, cv

def run_prog_sim(qvm:CPUQVM, prog:QProg, shot:int=1000, log:bool=True):
  res = qvm.run_with_configuration(prog, shot=shot)
  res = {k: v / sum(list(res.values())) for k, v in res.items()}
  if log: print(res)
  return res


def get_basis_encoder(qv:List[Qubit], x:List[float]) -> QCircuit:
  encoder = Encode()
  encoder.basic_encode(qv, x)
  return encoder.get_circuit()

def get_angle_encoder(qv:List[Qubit], x:List[float]) -> QCircuit:
  encoder = Encode()
  encoder.angle_encode(qv, x)
  return encoder.get_circuit()

def get_amplitude_encoder(qv:List[Qubit], x:List[float]) -> QCircuit:
  encoder = Encode()
  encoder.amplitude_encode(qv, x)
  return encoder.get_circuit()

# Hardware-Efficient Ansatz (RX-RY + CNOT)
def get_HAE(qv:List[Qubit], params:List[float], n_rep:int=3) -> QCircuit:
  nq = len(qv)
  vqc = QCircuit()
  p = 0
  for r in range(n_rep):
    for q in range(nq):
      vqc << RX(qv[q], params[p]) ; p += 1
      vqc << RY(qv[q], params[p]) ; p += 1
    if r < n_rep - 1:
      for q in range(nq - 1):
        vqc << CNOT(qv[q], qv[q+1])
  return vqc


def train(dataset:List, get_circuit:Callable, nq:int=3, maxiter=100, shot:int=3000) -> List[float]:
  def func(params:np.ndarray) -> float:
    nonlocal n_iter
    n_iter += 1

    losses = []
    for X, Y in dataset:
      qvm, qv, cv = get_qvm_sim(nq)
      prog = get_circuit(qv, cv, X, params)
      res = run_prog_sim(qvm, prog, shot=shot, log=False)
      probs = np.asarray([res.get(k, 0) for k in [bin(i)[2:].rjust(nq, '0') for i in range(nq**2)]])
      loss = ((probs - Y) ** 2).mean()
      losses.append(loss)
    loss_avg = sum(losses) / len(losses)
    if n_iter % 10 == 0:
      print(f'>> [{n_iter}/{maxiter}] loss: {loss_avg}')
    return loss_avg

  n_iter = 0
  params = np.random.uniform(low=-pi/32, high=pi/32, size=[nq*6])
  res = minimize(func, params, method='COBYLA', tol=1e-8, bounds=Bounds(-pi, pi), options={'maxiter': maxiter, 'disp': True})
  print('best f(x):', res.fun)
  print('best x:', res.x)
  return res.x

def infer(dataset:List, get_circuit:Callable, params:List[float], name:str, nq:int=3, shot:int=1000, use_real:bool=False):
  log_fp = LOG_PATH / (name + '.json')
  db = load_db(log_fp)
  try:
    if use_real:
      qvm, qv, cv = get_qvm(nq)
    else:
      qvm, qv, cv = get_qvm_sim(nq)
    for X, Y in dataset:
      prog = get_circuit(qv, cv, X, params)
      if use_real:
        res = qvm.real_chip_measure(prog, shot, real_chip_type.origin_72, is_amend=False, is_mapping=False, is_optimization=False)
      else:
        res = run_prog_sim(qvm, prog, shot=shot)
      probs = np.asarray([res.get(k, 0) for k in [bin(i)[2:].rjust(nq, '0') for i in range(nq**2)]])
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

def infer_clf(dataset:List, get_circuit:Callable, params:List[float], name:str, nq:int=3, shot:int=1000, use_real:bool=False):
  log_fp = LOG_PATH / (name + '.json')
  db = load_db(log_fp)
  tot, ok = 0, 0
  try:
    if use_real:
      qvm, qv, cv = get_qvm(nq)
    else:
      qvm, qv, cv = get_qvm_sim(nq)
    for X, Y in dataset:
      prog = get_circuit(qv, cv, X, params)
      if use_real:
        res = qvm.real_chip_measure(prog, shot, real_chip_type.origin_72, is_amend=False, is_mapping=False, is_optimization=False)
      else:
        res = run_prog_sim(qvm, prog, shot=shot)
      probs = np.asarray([res.get(k, 0) for k in [bin(i)[2:].rjust(nq, '0') for i in range(nq**2)]])
      db.append({
        'X': X,
        'res': dict(res),
        'probs': list(probs),
      })
      save_db(log_fp, db)

      truth = np.argmax(Y)
      pred = np.argmax(probs)
      print(f'>> truth: {truth}, pred: {pred}')
      if pred == truth: ok += 1
      tot += 1
      print(f'Acc:  {ok} / {tot} = {ok / tot:.3%}')
  except:
    print_exc()
  finally:
    save_db(log_fp, db)

  print(f'Acc:  {ok} / {tot} = {ok / tot:.3%}')
