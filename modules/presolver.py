import numpy as np
from pyballistics import ozvb_lagrange
from modules.solver import Cannon, Solver
from joblib import Parallel, delayed

def init_dict(dbname, wq=None, ro=None):
    
    if dbname is None:
        raise TypeError("empty name")
    d = 125*1e-3          # калибр м
    q = 7.05           # вес снаряда кг
    velocity_pm = 1700         # дульная скорость снаряда
    n_s = 1           # нарезное орудие
    max_pressure = 600*1e6    # максимальное давление Па
    tube_lenght = 5543*1e-3      # длина трубы     
    p_fors = 10*1e6  
    return  {
   'powders': 
    [
       {'omega': None if wq is None else wq*q,
       'dbname': dbname}
    ],
    'init_conditions': 
    {
       'q': q,
       'd': d,
       'W_0': None if ro is None else wq*q/ro,
       'phi_1': 1.0,
       'p_0': p_fors, 
       'n_S': n_s
    },
    'igniter': 
    {
       'p_ign_0': 5_000_000.0 #check it 
    },
  
    'meta_lagrange': 
    {
       'CFL': 0.9, 
       'n_cells': 300
    },
    'stop_conditions': 
    {
       'x_p': tube_lenght,
       'steps_max': 8000,   
       't_max': 0.08,
       'p_max': max_pressure,
       'v_p': velocity_pm,
    }
    }


def find_initial_point(initial_dict: dict, max_loop=1000):
    """
    This function generate random point and find initial point.
    Return <dict> with <bool> True if point has find else return empty <dict> with <bool> False
    """
    wq = np.random.rand(max_loop)*1.5 + 0.3
    ro = np.random.rand(max_loop)*1000 + 200
    wq0, ro0 = None, None
    success = False
    for i in range(max_loop):
        initial_dict['powders'][0]['omega'] = wq[i] * initial_dict['init_conditions']['q']
        initial_dict['init_conditions']['W_0'] = initial_dict['powders'][0]['omega']/ro[i]
        result = ozvb_lagrange(initial_dict)
        if result['stop_reason'] == 'x_p' or result['stop_reason'] == 'v_p' or result['stop_reason'] == 'p_max':
            wq0 = wq[i]
            ro0 = ro[i]
            success = True
            break
    return{
        'dbname': initial_dict['powders'][0]['dbname'],
        'wq': wq0,
        'ro': ro0,
        'reason': result['stop_reason'],
        'success': success,
    }  

def random_points(initial_dict: dict, center, bounds, max_loop=1000, v_d=1700):

   wq = (np.random.rand(max_loop)*2-1)*bounds[0] + center[0]
   ro = (np.random.rand(max_loop)*2-1)*bounds[1] + center[1]
   tabel = list()
   for i in range(max_loop):
      tabel.append(generate_point(wq[i], ro[i], initial_dict, v_d))
   return tabel

def random_points_multiproc(initial_dict: dict, center, bounds, max_loop=1000, v_d=1700, core=2):
   
   wq = (np.random.rand(max_loop)*2-1)*bounds[0] + center[0]
   ro = (np.random.rand(max_loop)*2-1)*bounds[1] + center[1]
   tabel = Parallel(n_jobs=core, verbose=4)(delayed(generate_point)(wq[i], ro[i], initial_dict, v_d) for i in range(max_loop))
   return tabel

def get_mass(result, initial_dict):

   diametr = initial_dict['init_conditions']['d']
   matrix_x = Solver.make_matrix(result, 'x')
   matrix_p = Solver.make_matrix(result, 'p')

   #add layre to 'p' for shape alignment with 'x'
   matrix_p = np.row_stack((matrix_p.T, matrix_p.T[-1])).T
   #calculate coordinate from 0 point
   l0 = np.abs(matrix_x[0][0])
   matrix_x += np.abs(matrix_x[0][0])
   cannon = Cannon(diametr, matrix_x, matrix_p, l0)
   return cannon.get_mass()

def generate_point(wq, ro, initial_dict, velocity_pm=1700):

   initial_dict['powders'][0]['omega'] = wq * initial_dict['init_conditions']['q']
   initial_dict['init_conditions']['W_0'] = initial_dict['powders'][0]['omega']/ro
   result = ozvb_lagrange(initial_dict)
   mass = None
   #if result['stop_reason'] == 'x_p' and result['layers'][-1]['u'][-1] < velocity_pm:
   #   reason='v_p'
   #else:
   #   reason = result['stop_reason']
   reason = result['stop_reason']
   if reason == 'v_p': # x_p
      try:
         mass = get_mass(result, initial_dict)
      except(ValueError):
         mass = None
         reason = "destroy"
   return {
      'wq': wq,
      'ro': ro,
      'reason': reason,
      'mass': mass,
      } 
