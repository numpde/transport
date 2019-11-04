
# RA, 2019-11-02

from time import time
import numpy as np
import pandas as pd

from contextlib import contextmanager
from scipy import sparse

@contextmanager
def timeit(description: str):
	print(f"{description}: Started")
	start = time()
	yield
	ellapsed_time = time() - start
	print(f"{description}: Done after {ellapsed_time * 1000}ms")


n = 2 ** 10

for m in [2 ** 10, 2 ** 20, 2 ** 30]:
	print("--")
	print(F"m = {m}")
	print("--")

	with timeit("making the matrix"):
		M = sparse.dok_matrix((m, n), dtype=float)
		M[np.random.choice(m, size=100), 0] = 1

	with timeit("matrix column"):
		w = sparse.dok_matrix(M[:, 0])
		print(list(w.items()))

	continue

	with timeit("matrix sum axis=0"):
		M.sum(axis=0)

	with timeit("matrix sum axis=1"):
		M.sum(axis=1)

	with timeit("making the dataframe"):
		df: pd.DataFrame
		df = pd.DataFrame.sparse.from_spmatrix(data=M)
		#.astype(pd.SparseDtype(dtype=float, fill_value=np.nan))

	with timeit("extracting column"):
		v = df[0]

	with timeit("df column as SparseArray"):
		v = pd.SparseArray(v, dtype=pd.SparseDtype())

	with timeit("math function map"):
		v.map(np.log)

	with timeit("sum of non null entries"):
		print(v.sum())

	# with timeit("to array"):
	# 	M.toarray()
