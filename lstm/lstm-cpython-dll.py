import ctypes
import numpy as np

# Load the shared library
lstm_lib = ctypes.CDLL('./lstm-weight-update.dll')  # Change the file extension to .dll

# Define the numpy array types as ctypes arrays
array_3d_float = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='CONTIGUOUS')
array_2d_float = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')

# Define the function prototype
lstm_lib.entry.argtypes = [array_3d_float, array_3d_float]
lstm_lib.entry.restype = ctypes.c_void_p

def call_entry(tensor_X, Y_h):
    lstm_lib.entry(tensor_X, Y_h)

tensor_X = np.linspace(0, 2*np.pi, 10).reshape(10, 1, 1).astype(np.float32)
Y_h = np.empty((1, 1, 10), dtype=np.float32)
call_entry(tensor_X, Y_h)

print(Y_h)
