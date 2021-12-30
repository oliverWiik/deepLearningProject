
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import os
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")



