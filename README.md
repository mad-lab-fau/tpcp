[![PyPI](https://img.shields.io/pypi/v/tpcp)](https://pypi.org/project/tpcp/)
[![Documentation Status](https://readthedocs.org/projects/tpcp/badge/?version=latest)](https://tpcp.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mad-lab-fau/tpcp/branch/main/graph/badge.svg?token=ZNVT5LNYHO)](https://codecov.io/gh/mad-lab-fau/tpcp)
[![Test and Lint](https://github.com/mad-lab-fau/tpcp/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/mad-lab-fau/tpcp/actions/workflows/test-and-lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/tpcp)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04953/status.svg)](https://doi.org/10.21105/joss.04953)

# tpcp - Tiny Pipelines for Complex Problems

A generic way to build object-oriented datasets and algorithm pipelines and tools to evaluate them.

Easily install `tpcp` via pip:
```bash
pip install tpcp
```

Or add it to your project with [poetry](https://python-poetry.org/):
```bash
poetry add tpcp
```

## Why?

Evaluating Algorithms - in particular when they contain machine learning - is hard.
Besides understanding required concepts (cross validation, bias, overfitting, ...), you need to implement the required 
steps and make them work together with your algorithms and data.
If you are doing something "regular" like training an SVM on tabular data, amazing libraries like [sklearn](https://scikit-learn.org), 
[tslearn](https://github.com/tslearn-team/tslearn), [pytorch](https://pytorch.org), and many others, have your back.
By using their built-in tools (e.g. `sklearn.evaluation.GridSearchCV`) you prevent implementation errors, and you are
provided with a sensible structure to organize your code that is well understood in the community.

However, often the problems we are trying to solve are not regular.
They are **complex**.
As an example, here is the summary of the method from one of our [recent papers](https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-021-00883-7):
- We have continuous multi-dimensional sensor recordings from multiple participants from a hospital visit and multiple days at home
- For each participant we have global metadata (age, diagnosis) and daily annotations
- We want to train a Hidden-Markov-Model that can find events in the data streams
- We need to tune hyper-parameters of the algorithm using a participant-wise cross validation
- We want to evaluate the final performance of the algorithm for the settings trained on the hospital data -> tested on home data and trained on home data -> tested on home data
- Using the same structure we want to evaluate a state-of-the-art algorithm to compare the results

None of the standard frameworks can easily abstract this problem, because here we have none-tabular data, multiple data 
sources per participant, a non-traditional ML algorithm, and a complex train-test split logic.

With `tpcp` we want to provide a flexible framework to approach such complex problems with structure and confidence.

## How?

To make `tpcp` easy to use, we try to focus on a couple of key ideas:

- Datasets are Python classes (think of [`pytorch.datasets`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), but more flexible) that can be split, iterated over, and queried.
- Algorithms and Pipelines are Python classes with a simple `run` and `optimize` interface, that can be implemented to fit any problem.
- Everything is a parameter and everything is optimization: In regular ML we differentiate *training* and *hyper-parameter optimization*.
  In `tpcp` we consider everything that modifies parameters or weights as an *optimization*.
  This allows to use the same concepts and code interfaces from simple algorithms that just require a grid search to optimize a parameter to neuronal network pipelines with hyperparameter tuning.
- Provide what is difficult, allow to change everything else:
  `tpcp` implements complicated constructs like cross validation and grid search and, whenever possible, tries to catch obvious errors in your approach.
  However, for the actual algorithm and dataset you are free to do whatever is required to solve your current research question.

## Should you use tpcp?

### Datasets

**Yes** - the object-oriented Datasets have proven themselves to be a really nice and flexible way to encapsulate Datasets with data from multiple modalities.
There is a clear path of integrating lazy-loading, load-cashing, data filtering, or pre-processing on loading.
From our experience, even if you ignore all the other tpcp features, Datasets can greatly simplify how you interact with your data sources and can serve as a self-documenting API for your data.

[Learn more](https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html#datasets) 
([Examples](https://tpcp.readthedocs.io/en/latest/auto_examples/index.html#datasets))

Other projects using Datasets:
- [gaitmap-datasets](https://github.com/mad-lab-fau/gaitmap-datasets)
- [cold-face-test-analysis](https://github.com/mad-lab-fau/cft-analysis/tree/main/cft_analysis/datasets)
- [carwatch-analysis](https://github.com/mad-lab-fau/carwatch-analysis/tree/main/carwatch_analysis/datasets)

### Parameter Optimization and Cross Validation

**Maybe** - All parameter optimization features in tpcp exist to provide a unified API, in case other specific frameworks are too specialised.
In cases where all your algorithms can be abstracted by `sklearn`, `pytorch` (with the `skorch` wrapper), `tensorflow`/`Keras` (with the `scikeras` wrapper), 
or any other framework that provides a nice scikit-learn API, you will get all the features tpcp can provide with much less boilerplate by just using `sklearn` and `optuna` directly.
Even, if you need to implement completely custom algorithms, we would encourage you to see if you can emulate a sklearn-like API to make use of its vast ecosystem.

This will usually work well for all algorithms that can be abstracted by the fit-predict paradigm.
However, for more "traditional" algorithms with no "fit" step or complicated optimizations, the `run` (with optional `self_optimize`) API of tpcp might be a better fit.
So if you are using or developing algorithms across library domains, that don't all work well with a sklearn API, then **Yes**, tpcp is a good choice.

Learn more:
[General Concepts](https://tpcp.readthedocs.io/en/latest/guides/index.html#user-guides),
[Custom Algorithms](https://tpcp.readthedocs.io/en/latest/auto_examples/index.html#algorithms), 
[Parameter Optimization](https://tpcp.readthedocs.io/en/latest/auto_examples/index.html#parameter-optimization), 
[Cross Validation](https://tpcp.readthedocs.io/en/latest/auto_examples/index.html#validation)

## Citation

If you use `tpcp` in your research, we would appreciate a citation to our JOSS paper.
This helps us to justify putting time into maintaining and improving the library.

```
Küderle et al., (2023). tpcp: Tiny Pipelines for Complex Problems - 
A set of framework independent helpers for algorithms development and evaluation. 
Journal of Open Source Software, 8(82), 4953,
https://doi.org/10.21105/joss.04953
```

```bibtex
@article{
  Küderle2023, 
  doi = {10.21105/joss.04953},
  url = {https://doi.org/10.21105/joss.04953},
  year = {2023}, publisher = {The Open Journal},
  volume = {8}, number = {82}, pages = {4953},
  author = {Arne Küderle and Robert Richer and Raul C. Sîmpetru and Bjoern M. Eskofier},
  title = {tpcp: Tiny Pipelines for Complex Problems - 
           A set of framework independent helpers for algorithms development and evaluation},
  journal = {Journal of Open Source Software}
}
```

## Contribution

The entire development is managed via [GitHub](https://github.com/mad-lab-fau/tpcp).
If you run into any issues, want to discuss certain decisions, want to contribute features or feature requests, just 
reach out to us by [opening a new issue](https://github.com/mad-lab-fau/tpcp/issues/new/).

## Dev Setup

We are using [poetry](https://python-poetry.org/) to manage dependencies and 
[poethepoet](https://github.com/nat-n/poethepoet) to run and manage dev tasks.

To set up the dev environment *including* the required dependencies for using and developing on `tpcp` run the following
commands: 
```bash
git clone https://github.com/mad-lab-fau/tpcp
cd tpcp
poetry install -E optuna -E torch # This might take a while
```

Note, that this will fail at the moment for Python 3.11., as pytorch is not yet available for this version.
You can safely skip the `-E torch` flag if you don't need the pytorch features.
All tests that require pytorch will be skipped automatically.

Afterwards you can start to develop and change things.
If you want to run tests, format your code, build the docs, ..., you can run one of the following `poethepoet` commands

```
CONFIGURED TASKS
  format         
  lint           Lint all files with Prospector.
  check          Check all potential format and linting issues.
  test           Run Pytest with coverage.
  docs           Build the html docs using Sphinx.
  bump_version   
```

by calling

```bash
poetry run poe <command name>
````

If you installed `poethepoet` globally, you can skip the `poetry run` part at the beginning.
