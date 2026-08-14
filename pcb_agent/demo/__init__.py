"""Runnable demonstrations of the Fireworks layer.

`fireworks_loop` is the full agent loop with MongoDB and the real trainer
stubbed out, so the Fireworks integration can be exercised on its own.
`memory_store` is the stub, exposing Person 1's frozen interface.
"""

from .memory_store import InMemoryLessonStore

__all__ = ["InMemoryLessonStore"]
