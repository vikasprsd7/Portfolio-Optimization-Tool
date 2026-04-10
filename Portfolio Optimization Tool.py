import numpy as np
import pandas as pd
import torch

## Detect if a GPU is available, otherwise use CPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
