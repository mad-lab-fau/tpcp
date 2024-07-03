# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-03

Note: This is a major version bump, because we have quite substantial breaking changes. The 1.0 should signal that we
are now feature complete. Though the core APIs have been mostly stable for quite some time now.

### BREAKING CHANGE

- Instead of the (annoying) `mock_label` and `group_label` arguments, all functions that take a cv-splitter as input,
  can now take an instance of the new `DatasetSplitter` class, which elegantly handles grouping and stratification and
  also removes the need of forwarding the `mock_label` and `group_label` arguments to the underlying optimizer.
  The use of the `mock_label` and `group_label` arguments has been removed without depreciation.
  (https://github.com/mad-lab-fau/tpcp/pull/114)
- All classes and methods that "grid-search" or "cross-validate" like output (`GridSearch`, `GridSearchCv`, `cross_validate`, `validate`)
  have updated names for all their output attributes.
  In most cases the output naming has switched from a single underscore to a double underscore to separate the different
  parts of the output name to make it easier to programmatically access the output.
  (https://github.com/mad-lab-fau/tpcp/pull/117)

## [0.34.1] - 2024-07-02

### Fixed
- The torch hasher was not working at all. This is hopefully fixed now.
- The tensorflow clone method did not work. Switched to specialized implementation that hopefully works.


## [0.34.0] - 2024-06-28

### Added

- Dataset classes are now generic and allow you to provide the group-label tuple as generic. This allows for better type 
  checking and IDE support. (https://github.com/mad-lab-fau/tpcp/pull/113)

### Changed/Fixed

- The snapshot utilities are much more robust now and rais appropriate errors when the stored dataframes have 
  unsupported properties. (https://github.com/mad-lab-fau/tpcp/pull/112)

## [0.33.1] - 2024-06-14

- Only warning about global caching once.

## [0.33.0] - 2024-05-23

### Added

- ``custom_hash`` the internally used hashing method based on pickle is now part of the public API via ``tpcp.misc``.
- ``DummyOptimize`` allows to ignore the warning that it usually throws.

### Changed

- Relative large rework of the TypedIterator. We recommend to reread the example.

## [0.32.0] - 2024-04-17

- The snapshot plugin now supports a new command line argument `--snapshot-only-check` that will fail the test if no
  snapshot file is found. This is useful for CI/CD pipelines, where you want to ensure that all snapshots are up to 
  date.
- The snapshot plugin is now installed automatically when you install tpcp. There is no need to set it up in the conftest
  file anymore.

## [0.31.2] - 2024-02-01

### Fixed

- TypedIterator does not run into a RecursionError anymore, when attributes with the wrong name are accessed.

## [0.31.1] - 2024-02-01

### Fixed

- TypedIterator now skips aggregation when no values are provided

## [0.31.0] - 2024-01-31

## Changed

- The TypedIterator now has a new `results_` attribute and has improved typing to allow for better IDE integration.

## [0.30.3] - 2024-01-23

### Fixed

- Downgraded minimum version of sklearn to 1.2.0 to avoid version conflicts with other packages.

## [0.30.2] - 2024-01-23

### Changed

- Better typing and Docstrings for the new functions introduced in 0.30.0

## [0.30.0] - 2024-01-23

### Added

- Added a new `classproperty` that allows to define class level properties equivalent to `@property` for instances.
- Added a new `set_defaults` decorator that allows to modify the default values or a function or class init.

## [0.29.0] - 2023-12-19

### Added

- Added a new `hybrid_cache` that allows to cache in RAM and Disk at the same time.

## [0.28.0] - 2023-12-19

### Changed

- The minimal version of pandas was reduced to 1.3. It still seems to work with that minimal version and this avoids 
  version conflicts with other packages.

### Added

- Helper to perform global caching of algorithm actions.
  This can be helpful to cache results of algorithms that are deeply nested within other methods or algorithms that are
  called multiple times withing the same pipeline.
  (https://github.com/mad-lab-fau/tpcp/pull/103)
- Clone now supports recursive cloning of dicts.
  This allows the theoretical use of dictionaries as parameters.

### Removed

- The test that checks if all mutable defaults are wrapped in `CloneFactory` is now removed.
  This check is performed at runtime anyway.

## [0.27.0] - 2023-11-09

### Added

- The TypedIterator (introduced in 0.26.0) now hase a base class (BaseTypedIterator), that can be used to implement 
  custom iterators that can get custom inputs to the `iterate` method, that are then further processed before the actual
  iteration.

## [0.26.2] - 2023-11-05

### Fixed

- Now actually fixed the pytest registration of the testing modules.

## [0.26.1] - 2023-11-05

### Fixed

- The testing modules are now registered as pytest files, which should result in verbose assert statements, making 
  debugging easier.

## [0.26.0] - 2023-11-03

### Added

- TypedIterator (https://github.com/mad-lab-fau/tpcp/pull/100): A new helper that makes iterating over things and accumulating results much easier.

### Changed

- Improved typing of "safe" decorators (https://github.com/mad-lab-fau/tpcp/pull/100).
  This should fix wrong IDE typehints.
- Now using py39 type typehints

## [0.25.1] - 2023-10-25

### Fixed

- Ignored names in the testing mixin are now correctly ignored both-ways.
  I.e. it allows to document additional parameters as well, not just leave out parameters.

## [0.25.0] - 2023-10-24

### Added

- The Scorer class now has the ability to score datapoints in parallel.
  This can be enabled by setting the `n_jobs` parameter of the `Scorer` class to something larger than 1.
  (https://github.com/mad-lab-fau/tpcp/pull/95)
- The `PyTestSnapshotTest` class does now support comparing dataframes with datetime columns.
  (https://github.com/mad-lab-fau/tpcp/pull/97)
- The `validate` function was introduced to enable validation of an algorithm on arbitrary data without parameter 
  optimization.
  (https://github.com/mad-lab-fau/tpcp/pull/99)
- Fixed the bug that the functions `optimize` and `cross_validate` were crashing when `progress_bar` was deactivated.
- New example about caching.
  (https://github.com/mad-lab-fau/tpcp/pull/98)

### Changed

- In line with numpy and some other packages, we drop Python 3.8 support


## [0.24.0] - 2023-09-08

For all changes in this release see: https://github.com/mad-lab-fau/tpcp/pull/85

### Deprecated

- The properties `group` and `groups` of the `Dataset` class are deprecated and will be removed in a future
  release.
  They are replaced by the `group_label` and `group_labels` properties of the `Dataset` class.
  This renaming was done to make it more clear that these properties return the labels of the groups and not the 
  groups themselves.
- The `create_group_labels` method of the `Dataset` class is deprecated and will be removed in a future release.
  It is replaced by the `create_string_group_labels` method of the `Dataset` class.
  This renaming was done to avoid confusion with the new names for `groups` and `group`

### Added

- Added `index_as_tuples` method to the `Dataset` class.
  It returns the full index of the dataset as a list of named tuples regardless of the current grouping.
  This might be helpful to extract the label information of a datapoint, when `group` requires to handle multiple cases,
  as your code expects the dataset in different grouped versions.

### Changed

- **BREAKING CHANGE (with Deprecation)**: The `group` property of the `Dataset` class is now called `group_label`.
- **BREAKING CHANGE**: The `group_label` property now always returns named tuples of strings
  (even for single groups where it used to return strings!).
- **BREAKING CHANGE (with Deprecation)**: The `groups` property of the `Dataset` class is now called `group_labels`.
- **BREAKING CHANGE**: The `group_labels` property always returns a list of named tuples of strings
  (even for single groups where it used to return a list of strings!).
- **BREAKING CHANGE**: The parameter `groups` of the `get_subset` method of the `Dataset` class is now called
  `group_labels` and always expects a list of named tuples of strings.

## [0.23.0] - 2023-08-30

### Added
- We migrated some testing utilities from other libraries to tpcp and exposed some algorithm test helper
  that previously only existed in the tests folder via the actual tpcp API.
  This should make testing algorithms and pipelines developed with tpcp easier.
  These new features are now available in the `tpcp.testing` module.
  (https://github.com/mad-lab-fau/tpcp/pull/89)

## [0.22.1] - 2023-08-30

### Fixed
- The `safe_optimize` parameter of `GridSearchCV` is now correctly used during reoptimization.
  Before, it was only forwarded to the `Optimize` wrapper during the actual Grid-Search, but not during the final
  reoptimization.

## [0.22.0] - 2023-08-25

### Added

- Official support for tensorflow/keras. The custom hash function now manages tensorflow models explicitly.
  This makes it possible again to use the `make_action_safe` and `make_optimize_safe` decorators with algorithms and 
  pipelines that have tensorflow/keras models as parameters.
  (https://github.com/mad-lab-fau/tpcp/pull/87)
- Added a new example for tensorflow/keras models.
  (https://github.com/mad-lab-fau/tpcp/pull/87)

## [0.21.0]

YANKED RELEASE

## [0.20.1] - 2023-07-25

### Fixed

- Fixed regression introduced in 0.19.0, which resulted in optimizers not beeing correctly cloned per fold.
  In result, each CV fold would overwrite the optimizer object of the previous fold.
  This did not affect the reported results, but the returned optimizer object was not the one that was used to calculate
  the results.

## [0.20.0] - 2023-07-24

### Changed

- **BREAKING CHANGE**: The way how all Optuna based optimizer work has been changed.
  Instead of passing a function, that returns a study, you now need to pass a function that returns the parameters of a
  study.
  Creating the study is now handled by tpcp internally to avoid issues with multiprocessing.
  This results in two changes.
  The parameter name for all optuna pipelines has changed from `create_study` to `get_study_params`.
  Further, the expected call signature changed, as `get_study_params` now gets a seed as argument.
  This seed should be used to initialize the random number generator of the sampler and pruner of a study to ensure
  that each process gets a different seed and sampling process.
  (https://github.com/mad-lab-fau/tpcp/pull/80)
  
  To migrate your code, you need to change the following:
  
  OLD:

  ```python
  def create_study():
      return optuna.create_study(sampler=RandomSampler(seed=42))
  
  OptunaSearch(..., create_study=create_study, ...)
  ```
  
  NEW:

  ```python
  def get_study_params(seed: int):
      return dict(sampler=RandomSampler(seed=seed))
  
  OptunaSearch(..., get_study_params=get_study_params, random_seed=42, ...)
  ```
     

## [0.19.0] - 2023-07-06

### Added

- All optimization methods that do complicated loops (over parameters or CV-Folds) now raise new custom error messages
  (OptimizationError and TestError) if they encounter an error.
  These new errors have further information in which iteration of the loop the error occurred and should make it easier
  to debug issues.
- When a scorer fails, we now print the name (i.e. the group) of the datapoint that caused the error.
  This should make it easier to debug issues with the scorer.

### Changed

- We dropped support for joblib<0.13.0. due to some changes in the API. We only support the new API now, which allowed 
  us to simplify some of the multiprocessing code.

## [0.18.0] - 2023-04-13

### Fixed
- When `super().__init__()` is called before all parameters of the child class are initialized, we don't get an error 
  anymore.
  Now all classes remember their parameters when they are defined and don't try to access parameters that are not 
  defined in their own init.
  (https://github.com/mad-lab-fau/tpcp/pull/69)

### Changed
- Validation is now performed recursively on all subclasses. Note like before validation is still only performed once 
  per class.
  But with this change, we can also validate base classes that are not used directly.
  (https://github.com/mad-lab-fau/tpcp/pull/70)

### Added
- We validate now, if a child class implements all the parameters of its parent class.
  While not strictly necessary, this is a sign of bad design, if not done.
  It could also lead to issues with tpcps validation logic.
  (https://github.com/mad-lab-fau/tpcp/pull/70)
- It is now possible to hook into the validation and perform custom validation of classes.
  (https://github.com/mad-lab-fau/tpcp/pull/70)
- The dataset class now activly triggers validation and checks if the dataset subclass implements `groupby_cols` and 
  `subset_index`.


## [0.17.0] - 2023-03-24

### Added
- We now have a workaround for global configuration that should be passed to worker processes when using 
  multiprocessing.
  This is a workaround to a [joblib issue](https://github.com/joblib/joblib/issues/1071) and is quite hacky.
  If you want to use this feature with your own configs you can use `tpcp.parallel.register_global_parallel_callback`.
  If you need to write your own parallel loop using joblib, you need to use `tpcp.parallel.delayed` instead of
  `joblib.delayed`.
  (https://github.com/mad-lab-fau/tpcp/pull/65)

## [0.16.0] - 2023-03-21

### Changed
- We are now raising an explicit ValidationError, if any of the parameters of a class have a trailing underscore, as 
  this syntax is reserved for result objects.
  (https://github.com/mad-lab-fau/tpcp/pull/63)

### Added
- The Optuna search methods have new parameter called `eval_str_paras` that allows to automatically turn categorical 
  string parameters into python objects.
  This can be usefull, if you want to select between complex objects and not just strings in your parameter search.
  To use this in your subclasses, you need to wrap the use of `trial.params` with `self.sanitize_params(trial.params)`.
  (https://github.com/mad-lab-fau/tpcp/pull/64)

## [0.15.0] - 2023-02-07

### Added
- GridSearch and GridSearchCV now have the option to pick the parameters with the lowest score if desired.
  This is useful, if your metric represents an error and you want to pick the parameters that minimize the error.
  To do that, you can set the `return_optimized` parameter of these classes to the name of metric prefixed with a `-`.
  (e.g. `return_optimized="-rmse"`).
  (https://github.com/mad-lab-fau/tpcp/pull/61)
- A new Optimization Algorithm called `OptunaSearch`. This is a (nearly) drop-in replacement for `GridSearch` using 
  Optuna under the hood.
  It can be used to quickly implement parameter searches with different samplers for non-optimizable algorithms.
  (https://github.com/mad-lab-fau/tpcp/pull/57)

### Changed
- In this release we added multiple safe guards against edge cases related to non-deterministic dataset indices.
  Most of these changes are internal and should not require any changes to your code.
  Still, they don't solve all edge cases. Make sure your index is deterministic ;)
  (https://github.com/mad-lab-fau/tpcp/pull/62)
  - The index of datasets objects are now cached
    The first time `create_index` is called, the index is stored in `subset_index` and used for subsequent calls.
    This should avoid the overhead of creating the index every time (in particular if the index creation requires IO).
    It should also help to avoid edge cases, where `create_index` is called multiple times and returns different results.
  - When `create_index` of a dataset is called, we actually call it twice now, to check if the index is deterministic.
    Having a non-deterministic index can lead to hard to debug issues, so we want to make sure that this is not the case.
    It could still be that the index changes when using a different machine/OS (which is not ideal for reproducibility),
    but this should prevent most cases leading to strange issues.
  - Internally, the `_optimize_and_score` method now directly gets the subset of the dataset, instead of the indices of 
    the train and test set.
    This should again help to avoid issues, where the index of the dataset changes between calculating the splits and 
    actually retrieving the data.

## [0.14.0] - 2023-02-01

### Added
- Custom Aggregators can now use the `RETURN_RAW_SCORES` class variable to specify, if their raw input scores should be 
  returned.
  (https://github.com/mad-lab-fau/tpcp/pull/58)

### Fixed
- GridSearch and GridSearchCV now correctly handle custom aggregators that return scores with new names.
  (https://github.com/mad-lab-fau/tpcp/pull/58)
- When using the `create_group_labels` method on dataset with multiple groupby columns, the method returned a list of 
  tuples.
  This caused issues with `GroupKFold`, as the method internally flattens the list of tuples.
  To avoid this, the method now return a list of strings.
  The respective string is simply the string representation of the tuple that was returned before.
  (https://github.com/mad-lab-fau/tpcp/pull/59)
- The fix provided in 0.12.1 to fix hashing of objects defined in the `__main__` module was only partially working.
  When the object in question was nested in another object, the hashing would still fail.
  This is hopefully now fixed for good.
  (https://github.com/mad-lab-fau/tpcp/pull/60)

## [0.13.0] - 2023-01-11

### Changed
- Some improvements to the documentation

### Added
- Added an option to the optuna search to use multiprocessing using the suggestions made in 
  https://github.com/optuna/optuna/issues/2862 .
  This has not been extensively tested in real projects.
  Therefore, use with care and please report any issues you encounter.

### Deprecated
- Fully deprecated the `_skip_validation` parameter for base classes, which was briefly used for some old versions.

## [0.12.2] - 2022-12-14

### Fixed
- The previous for fixing hashing of objects defined in the `__main__` module was not working
  This should now be fixed.

## [0.12.1] - 2022-12-14

### Changed
- The `safe_run` method did unintentionally double-wrap the run method, if it already had a `make_action_safe` 
  decorator. This is now fixed.

### Fixed
- Under certain conditions hashing of an object defined in the `__main__` module failed.
  This release implements a workaround for this issue, that should hopefully resolve most cases.

## [0.12.0] - 2022-11-15

### Added

- Added the concept of the `self_optimize_with_info` method that can be implemented instead or in addition to the 
  `self_optimize` method.
  This method should be used when an optimize method requires to return/output additional information besides the main
  result and is supported by the `Optimize` wrapper.
  (https://github.com/mad-lab-fau/tpcp/pull/49)
- Added a new method called `__clone_param__` that gives a class control over how params are cloned.
  This can be helpful, if for some reason objects don't behave well with deepcopy.
- Added a new method called `__repr_parameters__` that gives a class control over how params are represented.
  This can be used to customize the representation of individual parameters in the `__repr__` method.
- Add proper repr for `CloneFactory`

## [0.11.0] - 2022-10-17

### Added

- Support for Optuna >3.0
- Example on how to use `attrs` and `dataclass` with tpcp
- Added versions for `Dataset` and `CustomOptunaOptimize` that work with dataclasses and attrs. 
- Added first class support for composite objects (e.g. objects that need a list of other objects as parameters).
  This is basically sklearn pipelines with fewer restrictions (https://github.com/mad-lab-fau/tpcp/pull/48).

### Changed

- `CustomOptunaOptimize` now expects a callable to define the study, instead of taking a study object itself. 
  This ensures that the study objects can be independent when the class is called as part of `cross_validate`. 
- Parameters are only validated when `get_params` is called. This reduces the reliance on `__init_subclass__` and that 
  we correctly wrap the init.
  This makes it possible to easier support `attrs` and `dataclass`

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
