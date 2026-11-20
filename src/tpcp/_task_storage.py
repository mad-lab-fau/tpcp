"""Transactional task records shared by worker coordinators."""

import sqlite3
import threading
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import astuple, dataclass
from typing import Protocol


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    status: str
    attempt: str
    expires: float
    started: float
    finished: float | None
    artifact: str | None


class TaskTransaction(Protocol):
    """Read and replace records under one serializable transaction."""

    def get(self, task_id: str) -> TaskRecord | None: ...
    def put(self, record: TaskRecord) -> None: ...


class TaskStorage(Protocol):
    """Roll back on exception and commit all changes on normal exit."""

    def transaction(self) -> AbstractContextManager[TaskTransaction]: ...


class _Records:
    def __init__(self, connection):
        self.connection = connection

    def get(self, task_id):
        row = self.connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return TaskRecord(*row) if row else None

    def put(self, record):
        self.connection.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?)", astuple(record))


_SCHEMA = """CREATE TABLE IF NOT EXISTS tasks (
task_id TEXT PRIMARY KEY, status TEXT, attempt TEXT, expires REAL,
started REAL, finished REAL, artifact TEXT)"""


@contextmanager
def _transaction(connection):
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        yield _Records(connection)


class SQLiteTaskStorage:
    """Coordinate local processes using a SQLite file."""

    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(_SCHEMA)

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.path, timeout=60)) as connection, _transaction(connection) as records:
            yield records


class InMemoryTaskStorage:
    """Share this instance to coordinate workers in the same process."""

    def __init__(self):
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.execute(_SCHEMA)
        self._lock = threading.Lock()

    @contextmanager
    def transaction(self):
        with self._lock, _transaction(self._connection) as records:
            yield records
