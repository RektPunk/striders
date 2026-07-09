<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=striders&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>

**Striders** is a surrogate-based explainable AI (XAI) library for efficient feature attribution. It uses landmark-based kernel approximations to reduce the computational cost of generating additive feature contributions, providing a scalable alternative to SHAP for tabular models.

## Installation
```bash
pip install striders
```

## Performance Benchmarking
| Dataset (Task) | Samples / Features | Metric | **TreeSHAP** | **Striders** | **Speed-up** |
| --- | --- | --- | --- | --- | --- |
| **CA Housing** (Reg.) | 20,640 / 8 | Execution Time | 22.1948s | **0.3927s** | **56.5x** 🚀 |
|  |  | Fidelity ($R^2$) | - | **0.9081** |  |
|  |  | Correlation | - | **0.9490** |  |
| **Credit Default** (Clf.) | 30,000 / 23 | Execution Time | 47.0008s | **2.4718s** | **19.0x** 🚀 |
|  |  | Fidelity ($R^2$) | - | **0.9776** |  |
|  |  | Correlation | - | **0.9429** |  |

Reproducibility: You can run directly in:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RektPunk/striders/blob/main/examples/benchmark.ipynb)

## Acknowledgments & Citations
This is an **unofficial** implementation based on the principles described in:

```bibtex
@article{ko2025stride,
  title={STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings},
  author={Ko, Chaeyun},
  journal={arXiv preprint arXiv:2509.09070},
  year={2025}
}
```
