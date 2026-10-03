"""Shared timing: derived from the voiceover word timings."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build"))
TIM = json.load(open(os.path.join(BUILD, "timings.json")))
L = TIM["lines"]

def ls(i): return L[i][0]["t0"]          # line start
def wt(i, j): return L[i][j]["t0"]       # word start

OUTRO = 2.9
DUR = round(TIM["duration"] + OUTRO, 2)

EV = {
    "farm": 0.0,
    "you_are": ls(1),
    "wrong": wt(1, 2),
    "secret": ls(2) - 0.05,
    "drop": wt(3, 2),                    # logo reveal on "donuthugobet.net"
    "donut": ls(4) - 0.05,
    "hugo": ls(5) - 0.05,
    "coin": ls(6) - 0.05,
    "multi": wt(6, 3) - 0.05,
    "balance": ls(7) - 0.05,
    "explode": wt(7, 4),
    "instant": ls(8) - 0.05,
    "instant2": wt(8, 2) - 0.05,
    "stop": ls(9) - 0.05,
    "stop_word": wt(9, 1),
    "goto": wt(9, 3) - 0.05,
    "final": wt(9, 6),
    "tagline": wt(9, 6) + 1.5,
    "end": DUR,
}
