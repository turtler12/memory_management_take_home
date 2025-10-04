# src/memory_scheduler.py
"""
Run temporal decay + adaptive compression on an interval.

Usage:
  export NEO4J_URI=...
  export NEO4J_USER=...
  export NEO4J_PASSWORD=...
  python -m src.memory_scheduler          # defaults
  # or:
  INTERVAL_SEC=300 HALF_LIFE_MIN=60 CUTOFF=0.2 W_RECENCY=0.6 W_POP=0.4 python -m src.memory_scheduler
"""

import os, time, signal, sys
from .graph_memory import GraphMemory

INTERVAL = int(os.environ.get("INTERVAL_SEC", "300"))       # 5 min default
HALF_LIFE = float(os.environ.get("HALF_LIFE_MIN", "60"))    # minutes
CUTOFF = float(os.environ.get("CUTOFF", "0.2"))
W_REC = float(os.environ.get("W_RECENCY", "0.6"))
W_POP = float(os.environ.get("W_POP", "0.4"))

_running = True
def _sigint(sig, frame):
    global _running
    _running = False
signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)

def main():
    print(f"[scheduler] every {INTERVAL}s | half_life={HALF_LIFE}m cutoff={CUTOFF} weights=({W_REC},{W_POP})")
    gm = GraphMemory()
    try:
        while _running:
            stats = gm.decay_and_compress(half_life_min=HALF_LIFE,
                                          w_recency=W_REC,
                                          w_popularity=W_POP,
                                          cutoff=CUTOFF)
            print(f"[scheduler] compressed={stats['compressed']} cutoff={stats['cutoff']}")
            for _ in range(INTERVAL):
                if not _running:
                    break
                time.sleep(1)
    finally:
        gm.close()
        print("[scheduler] stopped")

if __name__ == "__main__":
    main()
