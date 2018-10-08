from contrast.ELSH import ELSH


def standard(data, k, l, r, cc):
    elsh = ELSH(data.point_set, k, l, r)
    elsh.storage()
    ap = elsh.query(data.query_indices, data.result_indices, cc)
    print("ELSH:", ap)
