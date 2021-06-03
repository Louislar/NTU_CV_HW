import torch
import os
print(torch.cuda.is_available())
print(torch.cuda.device_count())

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
