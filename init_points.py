import pandas as pd
import joblib
from modules import presolver
from joblib import Parallel, delayed
from pyballistics import get_powder_names

tabel_init_point = list()
core = joblib.cpu_count() - 2

def step(name):
    return presolver.find_initial_point(presolver.init_dict(name))

tabel = Parallel(n_jobs=core, verbose=10)(delayed(step)(name) for name in get_powder_names())


tabel_init_point = pd.DataFrame(tabel)
tabel_init_point.to_csv("data/init_points.csv", index=False)
