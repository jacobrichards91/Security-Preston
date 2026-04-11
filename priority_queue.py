"""
priority_queue.py — Thread-safe newest-first priority queue for detection events.
"""

import heapq
import threading


class NewestFirstQueue:
    """Thread-safe priority queue: highest timestamp = processed first."""

    def __init__(self):
        self._heap = []          # (neg_ts, seq, item)
        self._lock = threading.Condition()
        self._seq  = 0
        self._all  = []          # parallel list for display (newest first)

    def put(self, item, ts):
        with self._lock:
            heapq.heappush(self._heap, (-ts, self._seq, item))
            self._seq += 1
            # rebuild display list sorted newest → oldest
            self._all = [i for (_, _, i) in sorted(self._heap)]
            self._lock.notify()

    def get(self):
        """Block until an item is available, return it (newest first)."""
        with self._lock:
            while not self._heap:
                self._lock.wait()
            _, _, item = heapq.heappop(self._heap)
            self._all = [i for (_, _, i) in sorted(self._heap)]
            return item

    def peek_all(self):
        """Return waiting items sorted newest → oldest (for display only)."""
        with self._lock:
            return list(self._all)

    def qsize(self):
        with self._lock:
            return len(self._heap)

    def task_done(self):
        pass  # compatibility shim
