from typing import Tuple

import numpy as np
from numpy.typing import NDArray

class Striders:
    def __init__(self, m_landmarks: int, lambda_reg: float, sigma: float) -> None:
        pass
    def fit(self, x: NDArray[np.float32], pred: NDArray[np.float32]) -> None:
        pass
    def explain(
        self, x: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        pass
