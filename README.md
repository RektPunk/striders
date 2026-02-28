<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=striders&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/striders/releases/latest">
    <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/striders.svg">
  </a>
  <a href="https://github.com/RektPunk/striders/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RektPunk/striders.svg">
  </a>
</p>

**Striders** is a lightning-fast, surrogate-based model explanations (XAI). It provides an efficient alternative to traditional SHAP by leveraging landmark-based kernel approximations. Striders implements a landmark-based approximation of the Shapley Kernel. By selecting representative landmarks, it reduces the complexity of the explanation process while maintaining high correlation with the true Shapley values.



## Installation

```bash
pip install striders
```

## Performance Benchmarking

| Dataset (Task) | Samples / Features | Metric | **TreeSHAP** | **Striders** | **Speed-up** |
| --- | --- | --- | --- | --- | --- |
| **CA Housing** (Reg.) | 20,640 / 8 | Execution Time | 6.4243s | **0.0784s** | **82.0x** 🚀 |
|  |  | Fidelity ($R^2$) | - | **0.9093** |  |
|  |  | Correlation | - | **0.9506** |  |
| **Credit Default** (Clf.) | 30,000 / 23 | Execution Time | 13.7760s | **0.4027s** | **34.2x** 🚀 |
|  |  | Fidelity ($R^2$) | - | **0.9776** |  |
|  |  | Correlation | - | **0.9428** |  |

Reproducibility: You can reproduce these results by running the [**script**](https://github.com/RektPunk/strides/tree/main/examples/benchmark.py).

## Acknowledgments & Citations

This is an unofficial implementation based on the principles described in:

```bibtex
@article{ko2025stride,
  title={STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings},
  author={Ko, Chaeyun},
  journal={arXiv preprint arXiv:2509.09070},
  year={2025}
}
```
