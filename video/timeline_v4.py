"""Timing for the v4 cut (server branding, ends on 'Join DC in bio')."""
from timeline_v3 import EV as _EV

DUR = 27.0
_BAR = 4 * 60 / 112
EV = dict(_EV)
EV["logo"] = 20.9
EV["cta"] = EV["beat"] + _BAR * 9          # bar line ~22.7 s: drums come back with the CTA
EV["drums2"] = EV["cta"]
EV["end"] = DUR
