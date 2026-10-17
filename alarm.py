"""
Audio Alert Module for Driver Drowsiness Detection
Provides non-blocking audio alerts when drowsiness is detected.
Uses Windows native winsound on Windows, or frequency sound / beep fallback.
"""

import threading
import sys
import time

class AlarmManager:
    def __init__(self, frequency=2500, duration_ms=400):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.is_active = False
        self._thread = None
        self._stop_event = threading.Event()

    def _sound_loop(self):
        while not self._stop_event.is_set():
            try:
                if sys.platform == "win32":
                    import winsound
                    winsound.Beep(self.frequency, self.duration_ms)
                else:
                    # ANSI bell fallback for Linux/macOS
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(self.duration_ms / 1000.0)
            except Exception as e:
                time.sleep(0.5)

    def start_alarm(self):
        """Start repeating alarm in a background thread if not already running."""
        if not self.is_active:
            self.is_active = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sound_loop, daemon=True)
            self._thread.start()

    def stop_alarm(self):
        """Stop repeating alarm."""
        if self.is_active:
            self.is_active = False
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)
