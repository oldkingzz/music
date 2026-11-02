"""Ceiling, NOT causal: told the song's true chord on every beat, sings the consonant chord tone nearest
its previous note. Shows how high 'music' can go when the chord is known; never ranked as a method."""
from theory import ALTO, ALTO_MID, CONSONANT, CT


class Oracle:
    def __init__(self, lookup):
        self.lookup = lookup                     # (tonic, K, first notes) -> beats chords

    def start(self, meta):
        self.t, self.prev, self.z = meta['tonic'], None, None
        self.meta = meta

    def step(self, i, onset, midi, prev_dur):
        if self.z is None:
            self.z = self.lookup.get((self.t, self.meta['K'], tuple(self.meta['pos'][:16])), [])
        k = int(onset)
        z = self.z[k] if k < len(self.z) else -1
        ref = self.prev if self.prev is not None else ALTO_MID
        best = None
        for h in range(ALTO[0], min(ALTO[1], midi - 1) + 1):
            ok = (abs(midi - h) % 12) in CONSONANT and (z < 0 or (h - self.t) % 12 in CT[z])
            if ok and (best is None or abs(h - ref) < abs(best - ref)):
                best = h
        self.prev = best if best is not None else self.prev
        return best if best is not None else -1


def build(train):
    import bench3
    lookup = {}
    for split in ('val',):  # mini bundle: test split removed, oracle is val-only
        for songs in bench3.eval_sets(split).values():
            for s in songs:
                lookup[(s['tonic'], s['K'], tuple(b['pos'] for b in s['beats'][:16]))] = [b['z'] for b in s['beats']]
    return Oracle(lookup)
