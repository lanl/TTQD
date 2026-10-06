import unittest
import numpy

# set backend before importing other ttqd modules
from ttqd.linalg import backend, set_backend, _AVAILABLE_BACKENDS, _AVAILABLE_DEVICE
set_backend("torch")
from ttqd.network import mps

print("Available backends = ", _AVAILABLE_BACKENDS)
print("Available devices = ", _AVAILABLE_DEVICE)
print('backend is', backend)

# print(dir(numpy))

