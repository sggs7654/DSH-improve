import numpy as np
from contrast.PCAH import PCAH
from Data.GeneratedData import GeneratedData
from General.draw import draw

data = GeneratedData(seed=1)
w, t = PCAH(data=data.point_set, L=2)
d = draw()
d.new_hp(data=data, w=w, t=t)
