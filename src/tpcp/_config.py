"""TPCP runtime configuration for the current execution context."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from tpcp.parallel import _register_tpcp_parallel_side_channel

_action_safety_checks: ContextVar[bool] = ContextVar("tpcp_action_safety_checks", default=True)


def get_config() -> dict[str, bool]:
    """Return a copy of the runtime settings for the current execution context.

    Returns
    -------
    dict
        Current settings. ``action_safety_checks`` defaults to True.

    """
    return {"action_safety_checks": _action_safety_checks.get()}


def set_config(*, action_safety_checks: bool | None = None) -> None:
    """Set runtime configuration for the current execution context.

    Parameters
    ----------
    action_safety_checks
        If False, disable all checks performed by ``make_action_safe`` and ``Pipeline.safe_run``.
        This includes parameter mutation, return-type, and result checks.
        Cloning and optimization checks are unaffected.
        None leaves the current setting unchanged. Use :func:`config_context` for scoped changes.

    """
    if action_safety_checks is not None:
        _action_safety_checks.set(action_safety_checks)


@contextmanager
def config_context(*, action_safety_checks: bool | None = None) -> Iterator[None]:
    """Temporarily change runtime settings, restoring them even if the scope raises.

    Parameters
    ----------
    action_safety_checks
        Override whether action safety checks run. None retains the current setting.
        See :func:`set_config` for the affected checks.

    Notes
    -----
    Settings are local to the current thread and asynchronous execution context. Nested scopes restore the previous
    setting. Use TPCP's parallel helpers to propagate settings to worker tasks.

    """
    value = _action_safety_checks.get() if action_safety_checks is None else action_safety_checks
    token = _action_safety_checks.set(value)
    try:
        yield
    finally:
        _action_safety_checks.reset(token)


@contextmanager
def _restore_config(config: dict[str, bool]) -> Iterator[Callable[[], None]]:
    with config_context(**config):
        yield lambda: None


def _discard_worker_config(_data: None) -> None:
    """Keep worker-local configuration changes out of the parent context."""


_register_tpcp_parallel_side_channel("config", get_config, _restore_config, _discard_worker_config)
