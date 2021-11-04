[![PyPI](https://img.shields.io/pypi/v/tpcp)](https://pypi.org/project/tpcp/)
[![Documentation Status](https://readthedocs.org/projects/tpcp/badge/?version=latest)](https://tpcp.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mad-lab-fau/tpcp/branch/main/graph/badge.svg?token=ZNVT5LNYHO)](https://codecov.io/gh/mad-lab-fau/tpcp)
[![Test and Lint](https://github.com/mad-lab-fau/tpcp/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/mad-lab-fau/tpcp/actions/workflows/test-and-lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/tpcp)

# tpcp - Tiny Pipelines for Complex Problems

A generic way to build object-oriented datasets and algorithm pipelines and tools to evaluate them

```
pip install tpcp
```

## Why?

Evaluating Algorithms - in particular when they contain machine learning - is hard.
Besides understanding required steps (Cross-validation, Bias, Overfitting, ...), you need to implement the required 
concepts and make them work together with your algorithms and data.
If you are doing something "regular" like training an SVM on tabulary data, amazing libraries like [sklearn](https://scikit-learn.org), 
[tslearn](https://github.com/tslearn-team/tslearn), [pytorch](https://pytorch.org), and many others, have your back.
By using their built-in tools (e.g. `sklearn.evaluation.GridSearchCV`) you prevent implementation errors, and you are
provided with a sensible structure to organise your code that is well understood in the community.

However, often the problems we are trying to solve are not regular.
They are **complex**.
As an example, here is the summary of the method from one of our [recent papers](https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-021-00883-7):
- We have continues multi-dimensional sensor recordings from multiple participants from a hospital visit and multiple days at home
- For each participant we have global metadata (age, diagnosis) and daily annotations
- We want to train Hidden-Markov-Model that can find events in the data streams
- We need to tune hyper-parameters of the algorithm using a participant-wise cross-validation
- We want to evaluate the final performance of the algorithm for the settings trained on the hospital data -> tested on home data and trained on home data -> tested on home data
- Using the same structure we want to evaluate a state-of-the-art algorithm to compare the results

None of the standard frameworks can easily abstract this problem, because we had none-tabular data, multiple data 
sources per participant, a non-traditional ML algorithm, and a complex train-test split logic.

With `tpcp` we want to provide a flexible framework to approach such complex problems with structure and confidence.

## How?

To make `tpcp` easy to use, we try to focus on a couple of key ideas:

- Datasets are Python classes (think `pytorch.datasets`, but more flexible) that can be split, iterated over, and queried
- Algorithms and Pipelines are Python classes with a simple `run` and `optimize` interface, that can be implemented to fit any problem
- Everything is a parameter and everything is optimization: In regular ML we differentiate *training* and *hyper-parameter optimization*.
  In `tpcp` we consider everything that modifies parameters or weights as an *optimization*.
  This allows to use the same concepts and code interfaces from simple algorithms that just require a gridsearch to optimize a parameter to neuronal network pipelines with hyperparameter Tuning
- Provide what is difficult, allow to change everything else:
  `tpcp` implements complicated constructs like cross validation and gridsearch and whenever possible tries to catch obvious errors in your approach.
  However, for the actual algorithm and dataset you are free to do, whatever is required to solve your current research question.