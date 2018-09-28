from contrast.PCAH import PCAH
from General.draw import draw
from General.storage import get_code
from General.query import query


def standard(data, L, cc):
    w, t = PCAH(data=data.point_set, L=L)
    if data.point_set.shape[1] == 2:
        d = draw()
        d.new_hp(data=data, w=w.transpose(), t=t)
    code = get_code(data.point_set, w, t)
    ap = query(data.point_set, w, t,
               data.query_indices,
               data.result_indices, code, cc)
    print("PCAH:", ap)