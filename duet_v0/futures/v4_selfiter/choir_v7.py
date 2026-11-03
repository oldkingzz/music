"""Choir rehearsal engine v7 (run choir_1102): choir_v6 for a large library. v6 decides the song from her opening
interval alone, which was enough for 2-3 songs; with the 100-song library (choir_songs_v8.json) 16 of 17 POP909 songs
sung offline were taken for another song that opens with the same interval (selftest_panel_v16).

v7 (ChoirMany): the start is v6's (provisional chord on her first held note, the opening interval picks a song and
sets the tempo), but every song whose opening interval fits stays a candidate. On each of her next notes, her distinct
notes since her first one are aligned (choir_v5's DP, anchored at the song's first note, in her key) to each
candidate's opening; candidates costing PRUNE more than the best drop out, and when another candidate beats the song
being sung by SWITCH the choir switches to it (parts re-written from the matched notes: tempo from her first note to
the latest matched one). After ID_WINDOW distinct notes, or one candidate left, the song is final.

  ChoirMany(songs, bpm, click_all=False)   songs = choir_songs_v8.json['songs'] (or a subset)
"""
import choir_v6 as CV6

ChoirStart = CV6.ChoirStart


class ChoirMany(ChoirStart):
    PRUNE, SWITCH, ID_WINDOW = 2.0, 0.8, 12

    def __init__(self, songs, bpm, click_all=False, log=lambda s: None):
        super().__init__(songs, bpm, click_all=click_all, log=log)
        self.cands, self.id_h0 = set(), None
        self.diag['switches'] = 0

    def _set_cands(self, h0, dh):
        self.id_h0 = h0
        self.cands = {sid for sid, ((b0, w0), (b1, w1)) in self.open.items()
                      if self.START_RATIO[0] <= dh / (w1 - w0) <= self.START_RATIO[1]} | {self.song_id}
        self.diag['candidates_at_start'] = len(self.cands)

    def _begin(self, sid, h0, h1, d0, d1):
        super()._begin(sid, h0, h1, d0, d1)
        self._set_cands(h0, h1['pitch'] - h0['pitch'])

    def _confirm(self, note):
        h0 = self.prov_note
        super()._confirm(note)
        if self.arr is not None and not self.provisional and self.id_h0 is not h0:
            self._set_cands(h0, note['pitch'] - h0['pitch'])

    def _on_note(self, note):
        super()._on_note(note)
        if self.arr is not None and not self.provisional and len(self.cands) > 1 and not self.stopped:
            self._reidentify()

    def _reidentify(self):
        hd, seen = [], False
        for n in self.hnotes:
            seen = seen or n is self.id_h0
            if not seen or n.get('junk') or (hd and abs(n['pitch'] - hd[-1]['pitch']) < 0.6):
                continue
            hd.append(n)
        if len(hd) < 3:
            return
        h = [n['pitch'] for n in hd]
        res = {}
        for sid in self.cands:
            d = self.dist[sid][:len(h) + 4]
            cost, pairs = self._align(h, [m for b, m in d], h[0] - d[0][1])
            res[sid] = (cost, pairs)
        best = min(res, key=lambda s: res[s][0])
        self.cands = {s for s in res if res[s][0] <= res[best][0] + self.PRUNE}
        cur = res.get(self.song_id)
        if best != self.song_id and (cur is None or cur[0] - res[best][0] >= self.SWITCH) and res[best][1]:
            i, j = res[best][1][-1]
            d = self.dist[best]
            if i > 0 and j > 0 and d[j][0] > d[0][0]:
                self._silence()
                self._begin(best, hd[0], hd[i], d[0], d[j])
                self.cands = {s for s in res if res[s][0] <= res[best][0] + self.PRUNE}
                self.diag['switches'] += 1
                self.log(f'switch to {best} after {len(hd)} notes')
        if len(hd) >= self.ID_WINDOW:
            self.cands = {self.song_id}
        self.diag['id_final_notes'] = len(hd) if len(self.cands) == 1 else None
