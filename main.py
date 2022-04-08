import copy
import numpy as np
from modules.solver import Solver
from pyballistics import ozvb_lagrange, get_db_powder, get_powder_names

d = 125*1e-3          # калибр м
q = 4.85           # вес снаряда кг
velocity_pm = 1700         # дульная скорость снаряда
n_s = 1           # нарезное орудие
max_pressure = 1000*1e6    # максимальное давление Па
tube_lenght = 5543*10e-3      # длина трубы  

p_fors = 10*1e6      #давление форсирования (гладкоствольные 7-15 МПа)

initial_dict_static =  {
   'powders': 
    [
       {'omega': None, 'dbname': '14/7 В/А'}
    ],
  'init_conditions': 
    {
       'q': q,
       'd': d,
       'W_0': None ,
       'phi_1': 1.0,
       'p_0': p_fors, 
       'n_S': 1
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
       't_max': 0.05,
       'p_max': max_pressure,
        'v_p': velocity_pm,
    }
 }

powders_names = get_powder_names()
wq_0 = 0.9
ro_0 = 500


initial_dict = copy.deepcopy(initial_dict_static)
initial_dict['powders'][0]['omega'] = wq_0*q
initial_dict['init_conditions']['W_0'] = wq_0*q/ro_0
solv = Solver(initial_dict)
solv.solution()
print(solv.get_solution())