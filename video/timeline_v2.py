"""Timing for the v2 motion-design cut, derived from build_v2/timings.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build_v2"))
TIM = json.load(open(os.path.join(BUILD, "timings.json")))
L = TIM["lines"]

def ls(i): return L[i][0]["t0"]
def wt(i, j): return L[i][j]["t0"]
def we(i, j): return L[i][j]["t1"]

OUTRO = 3.0
DUR = round(TIM["duration"] + OUTRO, 2)

EV = {
    "hook": 0.0,
    "smarter": ls(1) - 0.22,
    "meet": ls(2) - 0.12,
    "drop": wt(2, 1),
    "donut": ls(3) - 0.15,
    "hugo": ls(4) - 0.12,
    "coin": ls(5) - 0.2,
    "multi": ls(6) - 0.12,
    "instant": ls(7) - 0.2,
    "instant2": ls(8),
    "end": ls(9) - 0.18,
    "final": wt(9, 3),
    "tagline": we(9, 3) + 0.25,
    "out": DUR,
}
