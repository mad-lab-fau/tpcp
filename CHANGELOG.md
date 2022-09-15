# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - unreleased

### Changed

- `CustomOptunaOptimize` now expects a callable to define the study, instead of taking a study object itself. 
  This ensures that the study objects can be independent when the class is called as part of `cross_validate`. 

## [0.10.0] - 2022-09-09

### Changed

- Reworked once again when and how annotations for tpcp classes are processed.
  Processing is now delayed until you are actually using the annotations (i.e. as part of the "safe wrappers").
  The only user facing change is that the chance of running into edge cases is lower and that `__field_annotations__` is
  now only available on class instances and not the class itself anymore.


## [0.9.1] - 2022-09-08

### Fixed
- Classes without init can now pass the tpcp checks

### Added
- You can nest parameter annotations into `ClassVar` and they will still be processed. 
  This is helpful when using dataclasses and annotating nested parameter values.

## [0.9.0] - 2022-08-11

This release drops Python 3.7 support!

### Added
- Bunch new high-level documentation
- Added submission version of JOSS paper

### Changed
- The `aggregate` methods of custom aggregators now gets the list of datapoints in additions to the scores.
  Both parameters are now passed as keyword only arguments.

## [0.8.0] - 2022-08-09

### Added
- An example on how to use the `dataclass` decorator with tpcp classes. (https://github.com/mad-lab-fau/tpcp/pull/41)
- In case you need complex aggregations of scores across data points, you can now wrap the return values of score 
  functions in custom `Aggregators`.
  The best plac eto learn about this feature is the new "Custom Scorer" example.
  (https://github.com/mad-lab-fau/tpcp/pull/42)
- All cross_validation based methods now have a new parameter called `mock_labels`.
  This can be used to provide a "y" value to the split method of a sklearn-cv splitter.
  This is required e.g. for Stratified KFold splitters.
  (https://github.com/mad-lab-fau/tpcp/pull/43)

### Changed
- Most of the class proccesing and sanity checks now happens in the init (or rather a post init hook) instead of during 
  class initialisation.
  This increases the chance for some edge cases, but allows to post-process classes, before tpcp checks are run.
  Most importantly, it allows the use of the `dataclass` decorator in combination with tpcp classes.
  For the "enduser", this change will have minimal impact.
  Only, if you relied on accessing special tpcp class parameters before the class (e.g. `__field_annotations__`) was 
  initialised, you will get an error now.
  Other than that, you will only notice a very slight overhead on class initialisation, as we know need to run some 
  basic checks when you call the init or `get_params`.
  (https://github.com/mad-lab-fau/tpcp/pull/41)
- The API of the Scorer class was modified.
  In case you used custom Scorer before, they will likely not work anymore.
  Further, we removed the `error_score` parameter from the Scorer and all related methods, that forwarded this parameter
  (e.g. `GridSearch`).
  Error that occur in the score function will now always be raised!
  If you need special handling of error cases, handle them in your error function yourself (i.e. using try-except).
  This gives more granular control and makes the implementation of the expected score function returns much easier on 
  the `tpcp` side.
  (https://github.com/mad-lab-fau/tpcp/pull/42)

## [0.7.0] - 2022-06-23

### Added

- The `Dataset` class now has a new parameter `group`, which will return the group/row information, if there is only a 
  single group/row left in the dataset.
  This parameter returns either a string or a namedtuple to make it easy to access the group/row information.
- The `Dataset.groups` parameter now returns a list of namedtuples when it previously returned a list of normal tuples.
- New `is_single_group` and `assert_is_single_group` methods for the `Dataset` class are added.
  They are shortcuts for calling `self.is_single(groupby_cols=self.groupby_cols)` and 
  `self.assert_is_single(groupby_cols=self.groupby_cols)`.

### Removed

- We removed the `OptimizableAlgorithm` base class, as it is not really useful.
  We recommend implementing your own base class or mixin if you are implementing a set of algorithms that need a normal
  and an optimizable version. 

## [0.6.3] - 2022-05-31

- It is now possible to use namedtuples as parameters and they are correctly cloned
  (https://github.com/mad-lab-fau/tpcp/issues/39)

## [0.6.2] - 2022-04-21

- The poetry lockfiles are now committed to the repository.
- Some dependencies are updated


## [0.6.1] - 2022-04-05

### Changed

- Fixed bug with tensor hashing (https://github.com/mad-lab-fau/tpcp/pull/37)
- Fixed an issue with memoization during hashing (https://github.com/mad-lab-fau/tpcp/pull/37)
- Fixed an issue that the `safe_optimize_wrapper` could not correctly detect changes to mutable objects.
  This is now fixed by pre-calculating all the hashes. (https://github.com/mad-lab-fau/tpcp/pull/38)

## [0.6.0] - 2022-04-04

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
  (https://github.com/mad-lab-fau/tpcp/pull/34)

### Changed

- The return type for the individual values in the `Scorer` class is not `List[float]` instead of `np.ndarray`.
  This also effects the output of `cross_validate`, `GridSearch.gs_results_` and `GridSearchCV.cv_results_`
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- `cf` now has "faked" return type, so that type checkers in the user code, do not complain anymore.
  (https://github.com/mad-lab-fau/tpcp/pull/29)
- All TypeVar Variables are now called `SomethingT` instead of `Something_` (https://github.com/mad-lab-fau/tpcp/pull/34)

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
