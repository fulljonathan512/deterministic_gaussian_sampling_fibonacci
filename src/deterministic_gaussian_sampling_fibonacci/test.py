"""Manual check script: compare NumPy vs torch Gaussian sampling on grids.

This script does NOT use pytest. You can run it in two ways:

1. From the *package directory* (this file's directory)::

       cd d:\\02_Studium\\03_Bachelor_Informatik_KIT\\01_Module\\Semester_VIII\\Bachelorarbeit\\dgs_fibonacci\\src\\deterministic_gaussian_sampling_fibonacci
       python run_sample_gaus_torch_check.py

2. From the project root as a module::

       cd d:\\02_Studium\\03_Bachelor_Informatik_KIT\\01_Module\\Semester_VIII\\Bachelorarbeit\\dgs_fibonacci
       python -m deterministic_gaussian_sampling_fibonacci.run_sample_gaus_torch_check

It will print, for several dimensions and sequence types, the empirical
means and covariances of the samples produced by
``sample_gaussian_fibonacci`` (NumPy) and
``sample_gaussian_fibonacci_torch`` (torch), plus the differences and a
"close?" summary.
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from sample_gaus import sample_gaussian_fibonacci
from sample_gaus_torch import sample_gaussian_fibonacci_torch

	
FIB_TYPES = ["Fibonacci"]
FIB_DIMS = [2, 3, 4, 5, 6]

# Path where all check_one results will be stored. We keep it in the same
# directory as this script so both this file and the plotting utility can
# easily access it.
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "check_results.pkl")

# In-memory collection of results; written out at the end of main().
_CHECK_RESULTS = []


def check_one(dim: int, type: str, sample_count: int) -> None:
	"""Check one (dim, type, sample_count) configuration.

	Prints the empirical mean and covariance from NumPy and torch
	implementations, plus their differences and closeness flags.
	"""
	print(f"Checking dim={dim}, type={type}, sample_count={sample_count}")

	mu = torch.rand(dim)
	A = torch.triu(torch.rand(dim,dim))
	# print(A)
	A = A + torch.eye(dim) * 1e-9
	cov_np = (A @ A.mT).numpy()

	orig, oIsNone  = sample_gaussian_fibonacci(mu.numpy(), cov_np,sample_count, type)
	new, nIsNone = sample_gaussian_fibonacci_torch(mu, A, sample_count,type)
	
	if(orig is None or new is None):
		equal = False
	else:
		equal = np.allclose(orig, new.numpy(), atol=1e-6, rtol=1e-3, equal_nan=True)
	
	print(f"------------------{equal}------------------")
	print(f"original is none: {oIsNone}")
	print(f"original is none: {nIsNone}")
	# if not equal:
	# 	print("Original:")
	# 	print(orig)
	# 	print("New:")
	# 	if(new is None):
	# 		print("None")
	# 	else:
	# 		print(new.numpy())

	# Store this run's data for later analysis/plotting. We keep everything
	# needed to reproduce the behaviour: A, the two sample arrays (or None),
	# and the flags/metadata.
	_CHECK_RESULTS.append(
		{
			"dim": dim,
			"type": type,
			"sample_count": sample_count,
			"A": A.detach().cpu().numpy(),
			"orig": None if orig is None else np.asarray(orig),
			"new": None if new is None else new.detach().cpu().numpy(),
			"oIsNone": bool(oIsNone),
			"nIsNone": bool(nIsNone),
			"equal": bool(equal),
		}
	)


def main() -> None:
	# General cases for all supported dims and types
	for type in FIB_TYPES:
		for dim in FIB_DIMS:
			for sample_count in range(1,10):
				check_one(dim, type, sample_count)

	# After all runs are finished, persist the collected results to disk so
	# that they can be analysed/visualised later.
	if _CHECK_RESULTS:
		with open(RESULTS_FILE, "wb") as f:
			pickle.dump(_CHECK_RESULTS, f)
		print(f"Saved {len(_CHECK_RESULTS)} check_one results to {RESULTS_FILE}")

if __name__ == "__main__":
	main()
