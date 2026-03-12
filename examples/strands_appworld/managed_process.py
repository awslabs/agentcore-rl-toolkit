import os
import select
import signal
import subprocess
import time


class ManagedProcess:
    """Wraps subprocess.Popen for lifecycle management with ready-pattern detection."""

    def __init__(self, cmd, ready_pattern="SERVER READY", startup_timeout=10.0):
        self.cmd = list(cmd)
        self.ready_pattern = ready_pattern
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None

    def start(self):
        if self.is_running():
            return

        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        if not self._wait_until_ready():
            self.stop()
            raise RuntimeError("ManagedProcess failed to start correctly")

    def _wait_until_ready(self) -> bool:
        """Return True if ready_pattern seen, False if crash/timeout."""
        assert self.proc is not None
        start_time = time.time()
        stdout = self.proc.stdout

        while True:
            if time.time() - start_time > self.startup_timeout:
                print("Startup timeout waiting for ready signal")
                return False

            if self.proc.poll() is not None:
                print(f"Process exited early with code {self.proc.returncode}")
                if stdout:
                    leftover = stdout.read()
                    if leftover:
                        print("Process output before exit:\n", leftover)
                return False

            if stdout is None:
                return True

            rlist, _, _ = select.select([stdout], [], [], 0.1)
            if not rlist:
                continue

            line = stdout.readline()
            if not line:
                continue

            if self.ready_pattern in line:
                return True

    def stop(self):
        if not self.is_running():
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        finally:
            self.proc = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
