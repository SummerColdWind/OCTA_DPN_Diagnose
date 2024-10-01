import pandas as pd
import os
import re
import shutil

def extract(title):
    items = title.split()[0].split('_')
    name, eye, type1, type2 = items[0], items[6], items[-2], items[-1]
    return name, eye, type1, type2


if not os.path.exists('clean'):
    os.makedirs('clean/Enface')
    os.makedirs('clean/OCTA')

for image in os.listdir('./raw/第一部分'):
    name, eye, type1, type2 = extract(image)
    print(name, eye, type1, type2)

    son_dir = f'clean/{type1[:-1]}/{name}_{eye}'
    if not os.path.exists(son_dir):
        os.makedirs(son_dir)
    shutil.copy(
        os.path.join('raw/第一部分', image),
        # os.path.join(son_dir, type1[:-1], f'{type2}.jpg')
        os.path.join(son_dir, f'{type2}.jpg')
    )









