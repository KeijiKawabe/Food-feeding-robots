# Pythonシェルで
import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.cuda version:", torch.version.cuda)   # PyTorchがリンクしてるCUDA
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("python:", sys.version)
