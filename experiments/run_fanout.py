"""
Fan-out runner: execute the independent experiments sequentially on the single
GPU (concurrent TabPFN fits would OOM on 8GB and corrupt memory metrics).

Order: Exp 2 (selection) -> Exp 3 (scaling) -> Exp 5 (shift) -> Exp 4 (choose k).
Each runs as its own subprocess; a failure in one is recorded and the queue
continues. Exit code always 0 so the launching process sees the full queue run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
QUEUE = [
    ("exp2_selection.py", {}),
    ("exp3_scaling.py", {}),
    ("exp5_shift.py", {}),
    ("exp4_choosing_k.py", {}),
]


def main():
    for script, env_extra in QUEUE:
        print("=" * 72)
        print(f"FANOUT: starting {script} at {time.strftime('%H:%M:%S')}")
        print("=" * 72, flush=True)
        import os
        env = dict(os.environ)
        env.update(env_extra)
        proc = subprocess.run(
            [sys.executable, str(HERE / script)],
            env=env, cwd=str(HERE),
            capture_output=True, text=True,
        )
        # Filter noisy warnings from the child output.
        for line in (proc.stdout or "").splitlines():
            low = line.lower()
            if any(t in low for t in ("hf_xet", "xet storage", "userwarning",
                                      "futurewarning", "warnings.warn")):
                continue
            print(line, flush=True)
        if proc.returncode != 0:
            print(f"FANOUT: {script} exited {proc.returncode}", flush=True)
            tail = (proc.stderr or "").splitlines()[-15:]
            print("STDERR tail:\n" + "\n".join(tail), flush=True)
        print(f"FANOUT: finished {script} at {time.strftime('%H:%M:%S')}\n",
              flush=True)
    print("FANOUT: all experiments complete.")


if __name__ == "__main__":
    main()
