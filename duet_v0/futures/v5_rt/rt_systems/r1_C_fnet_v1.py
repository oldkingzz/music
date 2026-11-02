"""r1_C_fnet_v1 (run v5rt_1101, round 1, idea C): HELPER, not a scored system (no build()). The learned joint
forecaster's features, network and a numpy copy of its forward pass, shared by the training script
(runs_mini/v5rt_1101/r1/C/train_fnet_C_v1.py) and the r1_C_* systems.

A small causal GRU reads her HEARD notes one by one (m3_seq_v1.feat layout: tonic-relative pc, octave, log previous
duration, bar position, beat phase, on-beat flag, meter). At a plan point q (a multiple of 0.5 beat) a query head
reads the GRU state after her last heard note (onset + L <= q), that note's features and the plan-point features
(bar position and half of q, time since her last heard onset, whether its end is heard, how long ago, its length)
and predicts, for the moment q:
  (a) on    P(a note of hers starts in [q, q + TOL))                                   (1 logit)
  (b) pc    her next note's TRUE pitch class relative to the tonic, given a note starts at q (12-way)
  (c) chord the chord at q: soft over the beats covered by the note starting at q (as m3_seq), else beat int(q) (7-way)
  (d) joint P(chord, pc) of the note starting at q, soft over its beats (84-way, index z * 12 + pc)
Inputs carry simulated 10 % mishearing in training; targets are TRUE pitches and chords from train labels only.
"""
import math

import numpy as np

NF = 24                # note features (m3_seq_v1.feat layout)
NQ = 14                # plan-point query features
HID = 96               # GRU size
TRUNK = 160            # query MLP width
L_TRAIN = 0.1          # beats: hearing latency used to build training examples
TOL = 0.01             # beats: a note "starts at q" if its onset is in [q, q + TOL) (pop909 sits 0.0003 after the grid)


def meter_index(M):
    return 0 if M == 2 else 1 if M == 3 else 2


def note_feat(pc, midi, prev_dur, pos, phase, M):
    """m3_seq_v1.feat, copied: one heard note."""
    f = np.zeros(NF, dtype=np.float32)
    f[pc] = 1.0
    f[12] = (midi - 66) / 12.0
    f[13] = 0.0 if prev_dur is None else math.log2(max(prev_dur, 1 / 16)) / 3.0
    f[14 + min(pos, 3)] = 1.0
    f[18] = math.sin(2 * math.pi * phase)
    f[19] = math.cos(2 * math.pi * phase)
    f[20] = 1.0 if phase < 1e-6 else 0.0
    f[21 + meter_index(M)] = 1.0
    return f


def query_feat(q, last, M, pos_q):
    """Plan point q. last = (onset, offset or None, midi) of her last heard note, or None before her first one;
    pos_q = bar position of beat int(q)."""
    f = np.zeros(NQ, dtype=np.float32)
    f[min(pos_q, 3)] = 1.0
    f[4] = 1.0 if q - math.floor(q) > 0.25 else 0.0
    f[5 + meter_index(M)] = 1.0
    if last is None:
        f[8] = 1.0
        return f
    on, off, _ = last
    dt = q - on
    f[9] = math.log2(max(dt, 1 / 16)) / 3.0
    f[10] = min(dt, 4.0) / 4.0
    if off is not None:
        f[11] = 1.0
        f[12] = math.log2(max(q - off, 1 / 16)) / 3.0
        f[13] = math.log2(max(off - on, 1 / 16)) / 3.0
    return f


def torch_net():
    """The torch module (imported lazily so the systems need numpy only at run time)."""
    import torch
    import torch.nn as nn

    class FNet(nn.Module):
        def __init__(self, hid=HID, trunk=TRUNK):
            super().__init__()
            self.inp = nn.Linear(NF, hid)
            self.gru = nn.GRU(hid, hid, batch_first=True)
            self.t1 = nn.Linear(hid + NF + NQ, trunk)
            self.t2 = nn.Linear(trunk, trunk)
            self.ha = nn.Linear(trunk, 1)
            self.hb = nn.Linear(trunk, 12)
            self.hc = nn.Linear(trunk, 7)
            self.hd = nn.Linear(trunk, 84)

        def encode(self, x, h=None):
            y, h = self.gru(torch.tanh(self.inp(x)), h)
            return y, h

        def head(self, s, x, qf):
            u = torch.relu(self.t1(torch.cat([s, x, qf], -1)))
            u = torch.relu(self.t2(u))
            return self.ha(u)[..., 0], self.hb(u), self.hc(u), self.hd(u)

    return FNet()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class NpNet:
    """numpy float64 copy of FNet's forward pass for one song, one note / one plan point at a time."""

    def __init__(self, sd):
        g = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v, dtype=np.float64)
             for k, v in sd.items()}
        self.Wi, self.bi = g['inp.weight'], g['inp.bias']
        self.Wih, self.Whh = g['gru.weight_ih_l0'], g['gru.weight_hh_l0']
        self.bih, self.bhh = g['gru.bias_ih_l0'], g['gru.bias_hh_l0']
        self.W1, self.b1, self.W2, self.b2 = g['t1.weight'], g['t1.bias'], g['t2.weight'], g['t2.bias']
        self.Wa, self.ba, self.Wb, self.bb = g['ha.weight'], g['ha.bias'], g['hb.weight'], g['hb.bias']
        self.Wc, self.bc, self.Wd, self.bd = g['hc.weight'], g['hc.bias'], g['hd.weight'], g['hd.bias']
        self.H = self.Whh.shape[1]

    def zero(self):
        return np.zeros(self.H)

    def step(self, h, x):
        """torch.nn.GRU's update (gates r, z, n) on the tanh input projection."""
        e = np.tanh(self.Wi @ x + self.bi)
        gi = self.Wih @ e + self.bih
        gh = self.Whh @ h + self.bhh
        H = self.H
        r = _sig(gi[:H] + gh[:H])
        z = _sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1.0 - z) * n + z * h

    def query(self, h, x, qf):
        """-> (P(onset at q), P(pc) 12, P(chord) 7, P(chord, pc) 7 x 12)."""
        u = np.maximum(self.W1 @ np.concatenate([h, x, qf]) + self.b1, 0.0)
        u = np.maximum(self.W2 @ u + self.b2, 0.0)
        p_on = float(_sig(self.Wa @ u + self.ba)[0])
        return (p_on, _softmax(self.Wb @ u + self.bb), _softmax(self.Wc @ u + self.bc),
                _softmax(self.Wd @ u + self.bd).reshape(7, 12))
