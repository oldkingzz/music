"""Baseline: the main chord HMM + planner, as planner.run(system='hmm')."""
from model import ChordHMM
from planner import run


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])

    def run(self, song, cfg):
        return run(song, self.model, cfg, 'hmm')


def build(train):
    return System(train)
