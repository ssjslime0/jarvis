"""Boot the Jarvis daemon and keep it alive long enough to confirm startup
and answer a real query through the engine (mirrors what the daemon does).

The daemon's stdin monitor requests shutdown when stdin hits EOF, so we hold
stdin open with a never-ending pipe. Audio (mic) is unavailable in this headless
environment, which is non-fatal: the daemon still boots, loads the model, and
the reply engine works.
"""

import sys
import os
import time
import subprocess
import io

# Keep our own stdout UTF-8 safe on Windows (emoji output).
if hasattr(sys.stdout, "buffer") and not getattr(sys, "frozen", False):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import types

sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["JARVIS_NO_AUDIO"] = "1"

    # Hold stdin open: a pipe we never write to and never close keeps the
    # daemon's stdin_monitor blocked on readline() instead of hitting EOF.
    r, w = os.pipe()
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "jarvis.daemon"],
        cwd=os.path.dirname(__file__),
        stdin=r,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )
    os.close(r)  # parent keeps only the write end open to hold the pipe

    # Stream daemon output for a while so the user sees it boot.
    start = time.time()
    alive_lines = []
    try:
        while time.time() - start < 45:
            raw = proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            alive_lines.append(line)
            if "Daemon started" in line:
                print("\n>>> Daemon booted. It is now running.\n", flush=True)
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        try:
            os.close(w)
        except Exception:
            pass

    booted = any("Daemon started" in l for l in alive_lines)
    return 0 if booted else 1


if __name__ == "__main__":
    sys.exit(main())
