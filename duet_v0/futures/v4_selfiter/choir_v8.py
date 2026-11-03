"""Choir rehearsal engine v8 (run choir_1102): choir_v7 with its song re-identification actually running. v7 found
her notes since the start by object identity (`n is self.id_h0`), but the start note choir_v6 hands over is not the
object kept in self.hnotes, so no note was ever collected and the opening interval alone kept deciding the song
(selftest_demo10_v1: Old MacDonald taken for v11, both open a 4th down; Going Home for Jingle Bells, both a minor 3rd
up). v8 takes her notes from the start note's time on. Everything else as v7.

  ChoirMany(songs, bpm, click_all=False)   songs = choir_songs_v9.json['songs']
"""
import choir_v7 as CV7


class ChoirMany(CV7.ChoirMany):

    def _reidentify(self):
        hd, t0 = [], self.id_h0['t'] - 1e-6
        for n in self.hnotes:
            if n['t'] < t0 or n.get('junk') or (hd and abs(n['pitch'] - hd[-1]['pitch']) < 0.6):
                continue
            hd.append(n)
        if len(hd) < 3:
            return
        h = [n['pitch'] for n in hd]
        res = {}
        for sid in self.cands:
            d = self.dist[sid][:len(h) + 4]
            res[sid] = self._align(h, [m for b, m in d], h[0] - d[0][1])
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
