import numpy as np
from numpy.typing import NDArray

class Striders:
    def __init__(self, num_bases: int, lambda_reg: float, sigma: float):
        pass
    def fit(self, x: NDArray[np.float32], pred: NDArray[np.float32]):
        pass
    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        pass
    def explain(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        pass
