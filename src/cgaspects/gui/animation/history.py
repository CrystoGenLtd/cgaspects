"""Session-only undo/redo history for the animation timeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("CGA:animation")


class TimelineHistory:
    """In-memory undo/redo stack of serialized timeline states.

    The stack lives only for the current session. If a path is given, the
    *current* state is additionally written there on every change as an
    autosave — the same format as "Save Animation…", so it can be recovered
    via "Load Animation…" after a crash.
    """

    def __init__(self, path: Optional[Path] = None, limit: int = 100):
        self._path = path
        self._limit = limit
        self._states: list[dict] = []
        self._index: int = -1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    def push(self, state: dict, autosave: bool = True) -> None:
        """Record a new state, discarding any redo tail.

        `autosave=False` records the state in the stack without writing the
        autosave file (used for the startup baseline, so a previous session's
        autosave isn't clobbered before the user has made any edit).
        """
        if 0 <= self._index < len(self._states) and self._states[self._index] == state:
            return  # no actual change
        del self._states[self._index + 1:]
        self._states.append(state)
        if len(self._states) > self._limit:
            self._states = self._states[-self._limit:]
        self._index = len(self._states) - 1
        if autosave:
            self._autosave()

    def undo(self) -> Optional[dict]:
        if not self.can_undo:
            return None
        self._index -= 1
        self._autosave()
        return self._states[self._index]

    def redo(self) -> Optional[dict]:
        if not self.can_redo:
            return None
        self._index += 1
        self._autosave()
        return self._states[self._index]

    def _autosave(self) -> None:
        if self._path is None or self._index < 0:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._states[self._index], f)
        except Exception:
            logger.exception("Failed to autosave timeline to %s", self._path)
