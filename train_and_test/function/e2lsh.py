from contrast.e2lsh import E2LSH


def standard(data, k, l, r, cc):
    e2lsh = E2LSH(data.point_set, k, l, r)
    e2lsh.storage()
    ap = e2lsh.query(data.query_indices, data.result_indices, cc)
    print("E2LSH:", ap)
