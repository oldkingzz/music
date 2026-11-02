"""Baseline: HMM + 'remember the song' (exp_ceiling.Repeat, last 2 beats must match)."""
from exp_ceiling import Repeat
from model import ChordHMM
from planner import run


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])

    def run(self, song, cfg):
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        return run(song, self.model, cfg, 'hmm', hint=Repeat(song, 2, D))


def build(train):
    return System(train)
