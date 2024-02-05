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
from pyqpanda import CPUQVM, QCloud, real_chip_type
from pyqpanda import QProg, Qubit, ClassicalCondition as CBit
from pyqpanda import H, I, X, Y, Z, RX, RY, RZ, CNOT, SWAP, Measure, measure_all
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


def get_qvm(nq:int) -> Tuple[QCloud, List[Qubit], List[CBit]]:
  qvm = QCloud()
  qvm.set_configure(72, 72)
  qvm.init_qvm(API_KEY, is_logged=True)
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
