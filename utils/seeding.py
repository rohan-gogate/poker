import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass