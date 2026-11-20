"""Resume a random trial and share preprocessing across trials
============================================================

Run this script twice with the same directory to replay the trial.
"""  # noqa: D205, D400

import asyncio
import random
from pathlib import Path

from joblib import Memory
from tpcp.parallel import CachedTask, SQLiteTaskStorage, Task, Worker


def square(value):
    return value * value


def draw(scale):
    return random.random() * scale


async def trial(directory, trial_id):
    directory.mkdir(exist_ok=True)
    async with Worker(
        Memory(directory / "cache", verbose=0),
        SQLiteTaskStorage(directory / "tasks.sqlite"),
    ) as worker:
        with worker.scope(trial_id):
            scale = await CachedTask(square)(3)
            return await Task(draw)(scale)


if __name__ == "__main__":
    print(asyncio.run(trial(Path("trial-results"), "trial-1")))
