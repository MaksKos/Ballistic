import pandas as pd
from modules import presolver
from pyballistics import ozvb_lagrange, get_db_powder, get_powder_names
from tqdm import tqdm

tabel_init_point = list()
core = 4

def step(name):
    return presolver.find_initial_point(presolver.init_dict(name))

for name in tqdm(get_powder_names()):
    tabel_init_point.append(step(name))

tabel = Parallel(n_jobs=core)(delayed(step)(name) for name in get_powder_names())


tabel_init_point = pd.DataFrame(tabel_init_point)
tabel_init_point.to_csv("data/init_points.csv", index=False)
