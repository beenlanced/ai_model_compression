### Prequisites

- Understanding of python

Talk about python version see LSTM

## Objective

The project contains the key elements:

- `Deep Learning` for neural networks building,
- `Git` (version control),
- `Jupyter` python coded notebooks,
- `Python` the standard modules,
- `PyTorch` Machine Learning framework to train our deep neural network,
- `Neural Network (NN)` to build process sequential data,
- `Tensors` mathematical objects that generalize scalars, vectors, and matrices into higher dimensions. A multi-dimensional array of numbers,
- `TensorBoard` visualization toolkit for TensorFlow that provides tools and visualizations for machine learning experimentation and,
- `uv` package management including use of `ruff` for linting and formatting

---

## Tech Stack

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.12` to be compatible with `PyTorch`.

You will just use

    ```bash
    uv add torch
    ```

Follow the requirements for [Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

- Make sure to use python versions `Python 3.12`
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.
- Windows Native Requires Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

## References

[Markdown](https://ashki23.github.io/markdown-latex.html)

[Affine Transformation](https://www.youtube.com/watch?v=AheaTd_l5Is)

[AI Model Compression-Quantizatin and Dequantization Explained with Examples and Mathematics ](https://medium.com/@0chandansharma/quantization-and-dequantization-explained-with-examples-and-mathematics-ecd48bdc55f1)

[Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

[Motivation behind the zero-point quantization and formula derivation, giving a clear interpretation of the “zero-point”](https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6)

### Write up

Quantization techniques are used to be able to use neural networks when there are constrains on memory or computation power, by trading some precision. Neural networks weights are usually trained in memory and computation-heavy floating point representations. With Quantization, we transform the weights into low precision representations (usually 8-bits), which use significantly less memory [Motivation]

In the case of Quantization, we start with a tensor whose values we want to project to the 8-bit range. Using 8-bits, we can only store up to 256 numbers. The INT8 representation typically used for quantization can store the 256 integer numbers between -128 and 127.

The range here refers to the range of **signed** bytes using [2's complement](http://en.wikipedia.org/wiki/2%27s_complement). Two's complement is the most common method of representing signed (positive, negative, and zero) integers on computers.
