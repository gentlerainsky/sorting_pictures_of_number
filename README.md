# Sorting Picture of Numbers with Algorithmic Representation Learning

## Introduction
In this project, we want to try to perform Sorting Hand Writing Numbers by using Algorithmic (Sorting) Representation Learning

## Structure

The model contains two parts.

1. Number Sorting Model

This model imitates (insertion) sorting algorithm over a list of 4 numbers.

2. CNN Model

This model try to learn multi digit MNIST using Model 1. as a loss function.

## Progress

At the moment, we can train (overfitly) a model to sort 4 digit number. But, we haven't succeed in training the CNN model yet.

## Reference

For [Algorithmic Representation Learning](https://github.com/deepmind/clrs)
```
@article{deepmind2022clrs,
  title={The CLRS Algorithmic Reasoning Benchmark},
  author={Petar Veli\v{c}kovi\'{c} and Adri\`{a} Puigdom\`{e}nech Badia and
    David Budden and Razvan Pascanu and Andrea Banino and Misha Dashevskiy and
    Raia Hadsell and Charles Blundell},
  journal={arXiv preprint arXiv:2205.15659},
  year={2022}
}
```

For [DiffSort](https://github.com/Felix-Petersen/diffsort)
```
@inproceedings{petersen2022monotonic,
  title={Monotonic Differentiable Sorting Networks},
  author={Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}

@inproceedings{petersen2021diffsort,
  title={Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision},
  author={Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}
```

For [Sorting Hand Writing Numbers Experiment](https://github.com/ermongroup/neuralsort)

```
@inproceedings{
grover2018stochastic,
title={Stochastic Optimization of Sorting Networks via Continuous Relaxations},
author={Aditya Grover and Eric Wang and Aaron Zweig and Stefano Ermon},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=H1eSS3CcKX},
}
```
