import pandas as pd
from modules import presolver
from pyballistics import ozvb_lagrange, get_db_powder, get_powder_names
from tqdm import tqdm

tabel_init_point = list()

for name in tqdm(get_powder_names()):
    tabel_init_point.append(presolver.find_initial_point(presolver.init_dict(name)))

tabel_init_point = pd.DataFrame(tabel_init_point)
tabel_init_point.to_csv("data/init_points.csv", index=False)
