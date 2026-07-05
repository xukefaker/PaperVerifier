from __future__ import annotations

import os
import select
import sys
import threading
import time


class CancelRequested(RuntimeError):
    pass


class ConsoleCancelWatcher:
    def __init__(self, *, key: str = "q") -> None:
        self.key = key.lower()
        self._cancelled = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios: list[int | bytes] | None = None

    def __enter__(self) -> "ConsoleCancelWatcher":
        if not sys.stdin.isatty():
            return self
        if os.name != "nt":
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                self._old_termios = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except Exception:
                self._old_termios = None
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._old_termios is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def check(self) -> None:
        if self._cancelled.is_set():
            raise CancelRequested("Canceled by user.")

    @property
    def requested(self) -> bool:
        return self._cancelled.is_set()

    def _watch(self) -> None:
        if os.name == "nt":
            import msvcrt

            while not self._stop.is_set():
                if msvcrt.kbhit() and msvcrt.getwch().lower() == self.key:
                    self._cancelled.set()
                    return
                time.sleep(0.1)
            return

        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable and sys.stdin.read(1).lower() == self.key:
                    self._cancelled.set()
                    return
            except Exception:
                return
