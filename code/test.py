import torch
import numpy as np

a = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
a = torch.Tensor(a)
print(a[a > 3] - 1)