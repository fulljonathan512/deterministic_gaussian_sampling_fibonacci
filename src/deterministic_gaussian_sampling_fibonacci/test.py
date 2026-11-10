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

import numpy as np
import torch
import matplotlib.pyplot as plt

from sample_gaus import sample_gaussian_fibonacci
from sample_gaus_torch import sample_gaussian_fibonacci_torch

	
FIB_TYPES = ["Fibonacci"]
FIB_DIMS = [2, 3, 4, 5, 6]


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

	orig = sample_gaussian_fibonacci(mu.numpy(), cov_np,sample_count, type)
	new = sample_gaussian_fibonacci_torch(mu, A, sample_count,type)
	
	if(orig is None or new is None):
		equal = False
	else:
		equal = np.allclose(orig, new.numpy(), atol=1e-6, rtol=1e-3, equal_nan=True)
	
	print(equal)
	if not equal:
		print("Original:")
		print(orig)
		print("New:")
		if(new is None):
			print("None")
		else:
			print(new.numpy())
	# print("-" * 60)
	# if(dim==2):
	# 	points_a = new
	# 	points_b = orig

	# 	plt.scatter(points_a[:, 0], points_a[:, 1], color="tab:blue", label="Array A")
	# 	plt.scatter(points_b[:, 0], points_b[:, 1], color="tab:orange", label="Array B")

	# 	plt.xlabel("x")
	# 	plt.ylabel("y")
	# 	plt.axis("equal")
	# 	plt.grid(True)
	# 	plt.legend()
	# 	plt.show()


def main() -> None:
	# General cases for all supported dims and types
	for type in FIB_TYPES:
		for dim in FIB_DIMS:
			for sample_count in range(1,10):
				check_one(dim, type, sample_count)

if __name__ == "__main__":
	main()
