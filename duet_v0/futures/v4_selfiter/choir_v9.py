"""Choir rehearsal engine v9 (run choir_1102): choir_v8, but the choir waits until it is sure of the song (Vincent
2026-11-03: "能不能就是晚一会儿再猜呢？我实际上不着急"). v8 sings from her first held note (the provisional chord) and
from her second note on the song the opening interval picks; when several songs open alike (Old MacDonald / v11,
Going Home / Jingle Bells) it sang the wrong one's chords for 3-5 s before switching. v9 schedules no part while the
start is provisional or more than one song is still a candidate; once one is left (or ID_WINDOW notes are heard) the
parts come in from the current beat (parts already over are skipped, the one sounding now starts at once).

  ChoirMany(songs, bpm, click_all=False)   songs = choir_songs_v9.json['songs']
"""
import choir_v8 as CV8


class ChoirMany(CV8.ChoirMany):
    SURE_MARGIN = None                                 # also sure when the song beats every other candidate by this

    def _sure(self):
        if self.arr is None or self.provisional:
            return False
        if len(self.cands) <= 1:
            return True
        c = getattr(self, 'id_costs', None)
        if self.SURE_MARGIN is None or not c or self.song_id not in c:
            return False
        return all(v - c[self.song_id] >= self.SURE_MARGIN for s, v in c.items() if s != self.song_id)

    def _align(self, h, d, off):
        out = super()._align(h, d, off)
        if self._costs is not None:                    # only while _reidentify runs (the follower aligns too)
            self._costs[tuple(d)] = out[0]
        return out

    _costs = None

    def _reidentify(self):
        self._costs = {}
        try:
            super()._reidentify()
            if self._costs:
                n = len(next(iter(self._costs)))
                self.id_costs = {sid: self._costs.get(tuple(m for b, m in self.dist[sid][:n])) for sid in self.cands}
                self.id_costs = {s: v for s, v in self.id_costs.items() if v is not None}
        finally:
            self._costs = None

    def _schedule(self):
        if self.arr is None or self.stopped:
            return
        if not self._sure():
            return                                     # parts wait (next_part kept): the backlog is written once sure
        if self.diag.get('sing_from') is None:
            sr = CV8.CV7.CV6.SR
            self.diag['sing_from'] = round(self.n / sr, 3)
            self.log(f'sure of {self.song_id}: choir comes in')
            while self.next_part < len(self.parts):    # parts already over: skip
                off = self.parts[self.next_part][1]
                if (self.t_of(off) - self.Lo - self.LEAD) * sr - int(self.GAP * sr) > self.choir.pos + int(0.05 * sr):
                    break
                self.next_part += 1
        super()._schedule()

    def _begin(self, sid, h0, h1, d0, d1):
        super()._begin(sid, h0, h1, d0, d1)
        self.diag['sing_from'] = None
