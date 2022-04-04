# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

### Added

- A new class to wrap the optimization framework [Optuna](https://optuna.readthedocs.io/en/stable/).
  `CustomOptunaOptimize` can be used to create custom wrapper classes for various Optuna optimizations, that play 
  nicely with `tpcp` and can be nested within tpcp operations. (https://github.com/mad-lab-fau/tpcp/pull/27)
- A new example for the `CustomOptunaOptimize` wrapper that explains how to create complex custom optimizers using
  `Optuna` and the new Scorer callbacks (see below) (https://github.com/mad-lab-fau/tpcp/pull/27)
- `Scorer` now supports an optional callback function, which will be called after each datapoint is scored.
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- Pipelines, Optimize objects, and `Scorer` are now `Generic`. This improves typing (in particular with VsCode), but 
  means a little bit more typing (pun intended), when creating new Pipelines and Optimizers
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- Added option for scoring function to return arbitrary additional information using the `NoAgg` wrapper
  (https://github.com/mad-lab-fau/tpcp/pull/31)
- (experimental) Torch compatibility for hash based comparisons (e.g. in the `safe_run` wrapper). Before the wrapper 
  would fail, with torch module subclasses, as their pickle based hashes where not consistent.
  We implemented a custom hash function that should solve this.
  For now, we will consider this feature experimental, as we are not sure if it breaks in certain use-cases.
  (https://github.com/mad-lab-fau/tpcp/pull/33)
- `tpcp.types` now exposes a bunch of internal types that might be helpful to type custom Pipelines and Optimizers.

### Changed

- The return type for the individual values in the `Scorer` class is not `List[float]` instead of `np.ndarray`.
  This also effects the output of `cross_validate`, `GridSearch.gs_results_` and `GridSearchCV.cv_results_`
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- `cf` now has "faked" return type, so that type checkers in the user code, do not complain anymore.
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- All TypeVar Variables are now called `SomethingT` instead of `Something_`

## [0.5.0] - 2022-03-15

### Added

- The `make_optimize_safe` decorator (and hence, the `Optimize` method) make use of the parameter annotations to check 
  that **only** parameters marked as `OptimizableParameter` are changed by the `self_optimize` method.
  This check also supports nested parameters, in case the optimization involves optimizing nested objects.
  (https://github.com/mad-lab-fau/tpcp/pull/9)
- All tpcp objects now have a basic representation that is automatically generated based on their parameters
  (https://github.com/mad-lab-fau/tpcp/pull/13)
- Added algo optimization and evaluation guide and improved docs overall
  (https://github.com/mad-lab-fau/tpcp/pull/26)
- Added examples for all fundamental concepts
  (https://github.com/mad-lab-fau/tpcp/pull/23)

## [0.4.0] - 2021-12-13

In this release the entire core of tcpc was rewritten and a lot of the api was reimagened.
This release also has no proper deprecation.
This repo will the wild west until the 1.0 release

There are a multitude of changes, that I do not remember all.
Here is the list of most important things:

- A lot of the import paths have changed. It is likely that you need to update some imports
- A new way to resolve the mutable default issue was introduced. It uses a explicit factory.
  To make that more clear, the name for mutable wrapper has changed from `mdf` -> `cf` or from `default` -> `CloneFactory`
- The overall class API got a lot slimmer. A bunch of methods like `get_attributes`, are now functions that can be 
  called with the class or instance as argument.
  Most methods were further renamed to represent there meaning in the context of `tpcp`.
  For example, `get_attributes` -> `get_results`
- The new concept of Parameter Annotation was added.
  You can add type annotations with the types `Parameter`, `HyperParameter`, `OptimizableParameter`, or `PureParameter`
  to a class parameter to indicate its role for the algorithm.
  This is in particularly helpful for optimizable Pipelines and algorithms.
  Wrapper methods can then use these annotations to run implementation checks and optimizations.
  The only method that uses these at the momement is `GridSearchCV`.
  It now allows to set `pure_parameters` to `True`, which will collect the list of all attributes annotated with
  `PureParameter` and use this information for performance optimization.
  Before, you needed to specify the list of pure parameters manually.
- Some core classes where renamed: `BaseAlgorithm` -> `Algorithm`, `SimplePipeline` -> `Pipeline`
- The decorators to make `run` and `self_optimize` safe are now reimagened and renamed to `make_run_safe` and 
  `make_optimize_safe`