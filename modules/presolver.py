import numpy as np
from pyballistics import ozvb_lagrange

def init_dict(dbname, wq=None, ro=None):
    
    if dbname is None:
        raise TypeError("empty name")
    d = 125*1e-3          # калибр м
    q = 4.85           # вес снаряда кг
    velocity_pm = 1700         # дульная скорость снаряда
    n_s = 1           # нарезное орудие
    max_pressure = 800*1e6    # максимальное давление Па
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
       'p_ign_0': 5000000.0 #check it 
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