#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/05

from scipy.optimize import minimize, Bounds
from utils import *

NAME = Path(__file__).stem

USE_REAL = True
MAX_ITER = 100
SHOT = 200
n_iter = 0
params = np.random.uniform(low=-pi/4, high=pi/4, size=[4])
target = np.asarray([1/3, 1/3, 1/3, 0])

def func(params:np.ndarray) -> float:
  global n_iter
  n_iter += 1
  if n_iter > MAX_ITER: raise ValueError(f'exceed MAX_ITER: {n_iter} > {MAX_ITER}')
  print(f'>> [{n_iter}/{MAX_ITER}]')

  if USE_REAL:
    qvm, qv, cv = get_qvm(2, log=False)
  else:
    qvm, qv, cv = get_qvm_sim(2)

  prog = QProg() \
    << RX(qv[0], params[0]) \
    << RY(qv[1], params[1]) \
    << CNOT(qv[0], qv[1]) \
    << RX(qv[0], params[2]) \
    << RY(qv[1], params[3]) \
    << measure_all(qv, cv)

  if USE_REAL:
    res = qvm.real_chip_measure(prog, SHOT, real_chip_type.origin_72, is_amend=False, is_mapping=False, is_optimization=False)
  else:
    res = run_prog_sim(qvm, prog, shot=SHOT)

  print(res)
  db.append({
    'iter': f'{n_iter}/{MAX_ITER}',
    'res': dict(res),
    'params': list(params),
  })
  save_db(log_fp, db)
  probs = np.asarray([res.get(k, 0) for k in ['00', '01', '10', '11']])
  loss = ((probs - target) ** 2).mean()
  print('loss:', loss)
  return loss


log_fp = LOG_PATH / (NAME + '.json')
db = load_db(log_fp)
try:
  minimize(func, params, method='COBYLA', tol=1e-4, bounds=Bounds(-pi, pi), options={'maxiter': MAX_ITER, 'disp': True})
except:
  print_exc()
finally:
  save_db(log_fp, db)
