"""Melody MIDI files of the 100-song choir library for Logic Pro (Vincent 2026-11-02: Logic plays the melody into the
mic instead of phone files or his own singing). One file per song: one track, the library melody at the song's own
key and tempo, program 53 (GM Voice Oohs), after one silent bar (time to get ready). Notes end 60 ms early so repeated
notes stay apart. Writes futures/live_test/midi_v1/<nnn>_<id>.mid and index.json (number, id, key, bpm, seconds).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py midi_export_v1 120 .venv/bin/python runs_mini/choir_1102/midi_export_v1.py
"""
import json
import os

import mido

ABS = '/Users/san/claude_sandbox/music/duet_v0'
LIB = os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v8.json')
OUT = os.path.join(ABS, 'futures/live_test/midi_v1')
TPB, GAP_S = 480, 0.06

if __name__ == '__main__':
    os.makedirs(OUT)
    with open(LIB) as fh:
        songs = json.load(fh)['songs']
    order = ['twinkle', 'v11', 'ode'] + sorted(k for k in songs if k not in ('twinkle', 'v11', 'ode'))
    index = []
    for n, sid in enumerate(order, 1):
        s = songs[sid]
        bpm = float(s['bpm'])
        gap = int(GAP_S * bpm / 60 * TPB)
        lead = 4 * TPB                                     # one silent bar
        ev = []
        for a, b, m in s['melody']:
            on, off = lead + int(round(a * TPB)), lead + int(round(b * TPB))
            ev += [(on, 1, m), (max(on + TPB // 8, off - gap), 0, m)]
        ev.sort(key=lambda e: (e[0], e[1]))                # note-offs before note-ons at the same tick
        tr = mido.MidiTrack([mido.MetaMessage('track_name', name=f'{n:03d} {sid}', time=0),
                             mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0),
                             mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0),
                             mido.Message('program_change', program=53, channel=0, time=0)])
        t = 0
        for tick, on, m in ev:
            tr.append(mido.Message('note_on' if on else 'note_off', note=int(m), velocity=96 if on else 0,
                                   channel=0, time=tick - t))
            t = tick
        mf = mido.MidiFile(ticks_per_beat=TPB)
        mf.tracks.append(tr)
        path = os.path.join(OUT, f'{n:03d}_{sid}.mid')
        mf.save(path)
        index.append({'n': n, 'id': sid, 'key': s['key'], 'bpm': round(bpm, 1),
                      'seconds': round((s['K'] + 4) * 60 / bpm, 1), 'file': os.path.basename(path)})
    with open(os.path.join(OUT, 'index.json'), 'x') as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)
    print(len(index), 'files in', OUT, flush=True)
