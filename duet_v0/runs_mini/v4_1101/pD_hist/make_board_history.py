"""Phase D (PROGRAM §5.2 board, §5.5 history): build report/board.txt and report/history.txt.
Reads only: runs_mini/v4_1101/nd/<name>/results/nd_<name>.json for each directory directly under nd/ (nothing
nested deeper), the last val row per system of futures/v3_nolatency/leaderboard_v3.jsonl, and
futures/v3_nolatency/results/nd_m4_oracle.json. Statuses are transcribed from state_R0..R4.json.
Without --write it only prints; with --write it creates both files with mode 'x' (refuses to overwrite)."""
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
ND = os.path.join(RUN, 'nd')
REPORT = os.path.join(RUN, 'report')
LB = os.path.join(ABS, 'futures', 'v3_nolatency', 'leaderboard_v3.jsonl')
ORACLE = os.path.join(ABS, 'futures', 'v3_nolatency', 'results', 'nd_m4_oracle.json')
DS = ['bach', 'folk', 'pop', 'pop909']

# ---------------------------------------------------------------- data
names = sorted(d for d in os.listdir(ND)
               if os.path.isdir(os.path.join(ND, d)) and os.path.isfile(os.path.join(ND, d, 'results', f'nd_{d}.json')))
ND_ = {n: json.load(open(os.path.join(ND, n, 'results', f'nd_{n}.json'))) for n in names}
LAST = {}
for line in open(LB):
    line = line.strip()
    if line:
        e = json.loads(line)
        if e.get('split') == 'val':
            LAST[e['system']] = e

# ---------------------------------------------------------------- status (from state_R0..R4.json)
GATE_ONLY = 'passed gate, not champion'
ST = {
    'r2_mix_v2': 'presentable champion (kept R2; E2 a=0.7)',
    'r3_style_mix_v3': 'best score, not presentable (kept R3; E2xE3 a=0.5)',
    'r2_gru_chooser_v1': 'passed keep, not champion (R2; E1 w_big 0.6)',
    'r2_style_hmm_v3': 'passed keep, not champion (R2; E3 4 styles)',
    'r2_gru_chooser_v2': GATE_ONLY + ' (R2; E1 w_big 0.8)',
    'r2_gru_chooser_v3': GATE_ONLY + ' (R2; E1 w_big 1.0)',
    'r2_mix_v1': GATE_ONLY + ' (R2; E2 a=0.5)',
    'r2_mix_v4': GATE_ONLY + ' (R2; E2 a=0.85 + w_big 0.8)',
    'r2_style_hmm_v1': GATE_ONLY + ' (R2; E3 3 styles)',
    'r2_style_hmm_v4': GATE_ONLY + ' (R2; E3 4 styles lam 1.5)',
    'r1_gru_chooser_v1': 'failed gate (R1; E1: folk big)',
    'r1_gru_chooser_v2': 'failed gate (R1; E1 one GRU: folk big)',
    'r2_mix_v3': 'failed gate (R2; E2 a=0.85: folk big)',
    'r2_style_hmm_v2': 'failed gate (R2; E3: folk +0.003)',
    'r3_style_mix_v1': 'failed gate (R3; E2xE3 a=0.7: headline)',
    'r3_style_mix_v2': 'failed gate (R3; E2xE3 4 styles: headline)',
    'r3_style_mix_v4': 'failed gate (R3; E2xE3 a=0.85: headline, folk, pop)',
    'r3_cover_fit_v1': 'dead idea (E4; R3 no gain)',
    'r3_cover_fit_v2': 'dead idea (E4; R3 no gain)',
    'r3_cover_fit_v3': 'dead idea (E4; R3 no gain)',
    'r3_cover_fit_v4': 'dead idea (E4; R3 no gain)',
    'r4_cover_fit_v1': 'dead idea (E4; R4 no gain)',
    'r3_true_cons_v1': 'dead idea (E5; R3 no gain)',
    'r3_true_cons_v2': 'dead idea (E5; R3 no gain)',
    'r3_true_cons_v3': 'dead idea (E5; R3 no gain)',
    'r3_true_cons_v4': 'dead idea (E5; R3 no gain)',
    'r4_true_cons_v1': 'dead idea (E5; R4 no gain)',
    'r4_true_cons_v2': 'dead idea (E5; R4 no gain)',
    'm1_filter_v5lo': 'baseline (old Tuesday model, starting champion)',
    'm3_seq_v5': 'baseline (3-GRU average + its own copying chooser)',
    'third_below': 'baseline (a third below her)',
}
for h in (0, 4, 16):
    ST[f'm3_seq_v5h{h}'] = f'history ablation (m3_seq_v5, last {h} beats)'
    ST[f'rD_hist_m1_h{h}_v1'] = f'history ablation (m1_filter_v5lo, last {h} beats)'
    ST[f'rD_hist_mix_h{h}_v1'] = f'history ablation (r2_mix_v2, last {h} beats)'
missing = [n for n in names if n not in ST]
assert not missing, f'no status for {missing}'


def row_bp(name):
    e = LAST.get(name)
    if e is None:
        return None, None, 'no bench3 row'
    return ([e['rows'][d]['big'] for d in DS], [e['rows'][d]['par'] for d in DS], ' '.join(e['flags']) or '-')


def line(rank, name, nd, big, par, flags, status):
    s = f"{rank:>3} {name:20s} {nd['headline_nd']:.4f} "
    s += ' '.join(f"{nd[d]['music_nd']:.4f}" for d in DS) + '  '
    s += ' '.join(f"{nd[d]['double']:.3f}" for d in DS) + '  '
    s += (' '.join(f'{v:.3f}' for v in big) if big else ' '.join(['  n/a'] * 4)) + '  '
    s += (' '.join(f'{v:.3f}' for v in par) if par else ' '.join(['  n/a'] * 4)) + '  '
    return s + f'{flags:5s}  {status}'


# ---------------------------------------------------------------- board.txt
B = []
B.append('Duet v4 board (Mac mini run v4_1101, written in Phase D on 2026-11-01): every system scored with rescore_nd under')
B.append('runs_mini/v4_1101/nd/, ranked by headline music_nd.  val only, tuned on val (this bundle has no test split).')
B.append('')
B.append('music_nd = share of her singing time (from beat 4) where Duet is consonant with her AND on a tone of the true chord AND not her note/octave; headline = mean of the 4 datasets.')
B.append('double   = share of her singing time where Duet sings her own note or its octave (copying her; music_nd counts it as a miss; keep rule allows <= 0.08).')
B.append('big      = share of Duet\'s note changes that leap more than 7 semitones (bench3; ROUGH flag above 0.10; keep rule: <= champion + 0.01 per dataset).')
B.append('par      = share of Duet\'s consecutive note pairs that move with her into parallel 5ths/octaves (bench3; ROUGH above 0.05; keep rule: <= champion + 0.01).')
B.append('flags    = bench3 flags: SLOW (> 2000 us per note), DOUBLING (double > 0.25), ROUGH (big > 0.10 or par > 0.05); "-" = none.')
B.append('Sources: music_nd and double from nd/<name>/results/nd_<name>.json; big, par and flags from the LAST val row of')
B.append('futures/v3_nolatency/leaderboard_v3.jsonl for that system; status from state_R0..R4.json.  Datasets: bach 17, folk 170, pop 300, pop909 40 val songs.')
B.append('')
B.append('                                   ---------- music_nd ----------  -------- double -------  --------- big ---------  --------- par ---------')
B.append('  # system               headline  bach   folk   pop    pop909    bach  folk  pop   p909   bach  folk  pop   p909   bach  folk  pop   p909   flags  status')
ranked = sorted(names, key=lambda n: -ND_[n]['headline_nd'])
for i, n in enumerate(ranked, 1):
    big, par, flags = row_bp(n)
    B.append(line(i, n, ND_[n], big, par, flags, ST[n]))
B.append('')
orc = json.load(open(ORACLE))
big, par, flags = row_bp('m4_oracle')
B.append(line('--', 'm4_oracle', orc, big, par, flags, 'ceiling, not causal, not ranked (told the true chord)'))
B.append('    (m4_oracle: music_nd/double from futures/v3_nolatency/results/nd_m4_oracle.json, big/par from its last val leaderboard row; both laptop, 2026-10-31.)')
B.append('')
B.append('Status legend (verdicts from state_R0..R4.json):')
B.append('  presentable champion        passed the full keep rule (PROGRAM §4.2) and is one chord guesser + m1_filter_v5lo\'s chooser (§4.3);')
B.append('                              the Tuesday model if Vincent passes its quiz, otherwise m1_filter_v5lo stays.')
B.append('  best score, not presentable passed the full keep rule against r2_mix_v2; its style-aware HMM is more than "the HMM" (strict §4.3).')
B.append('  kept                        passed the full keep rule and became champion in that round: only the two lines above (R2, R3).')
B.append('  passed keep, not champion   passed the full keep rule against m1_filter_v5lo (numbers, review, seed 2) but a higher candidate won the round.')
B.append('  passed gate, not champion   passed the numeric part of the keep rule only (not reviewed, no seed 2) and was below the round\'s winner.')
B.append('  failed gate                 failed the numeric keep rule (the failed check in brackets).')
B.append('  dead idea                   its idea (E4, E5) failed twice and is never retried.')
B.append('  baseline                    reference systems; m1_filter_v5lo is the old Tuesday model and the starting champion.')
B.append('  history ablation            Phase D: the system limited to her last H beats of melody (see history.txt).')
B.append('')
B.append(f'{len(ranked)} ranked rows = 28 scored loop candidates (R1 2, R2 11, R3 12, R4 3) + 3 baselines + 9 Phase D history ablations.')
B.append('The rank is by score only. Told the onset chord with m1\'s own chooser: 0.8403 (runs_mini/v4_1101/diag/diag_rt_v1.json, a diagnostic, not a system).')
B.append('Known label limit: the chord labels are the 7 basic triads of the major key, so 7th chords and the raised 7th in minor-key songs count as wrong.')
B.append('val only, tuned on val.')
board_txt = '\n'.join(B) + '\n'

# ---------------------------------------------------------------- history.txt
SYS = [
    ('m1_filter_v5lo (HMM chord filter + no-copy chooser; the old Tuesday model)',
     ['rD_hist_m1_h0_v1', 'rD_hist_m1_h4_v1', 'rD_hist_m1_h16_v1', 'm1_filter_v5lo']),
    ('m3_seq_v5 (3-GRU average + its own chooser, which copies her up to 0.18)',
     ['m3_seq_v5h0', 'm3_seq_v5h4', 'm3_seq_v5h16', 'm3_seq_v5']),
    ('r2_mix_v2 (presentable champion: b_HMM^0.3 * b_GRU^0.7, then m1_filter_v5lo\'s chooser)',
     ['rD_hist_mix_h0_v1', 'rD_hist_mix_h4_v1', 'rD_hist_mix_h16_v1', 'r2_mix_v2']),
]
WIN = ['current note only (H=0)', 'last 4 beats (H=4)', 'last 16 beats (H=16)', 'whole song']
H = []
H.append('Duet v4 history ablation in music_nd (PROGRAM §5.5): how much of her past melody does the real-time chord guess need?')
H.append('Mac mini run v4_1101, Phase D, 2026-11-01.  val only, tuned on val (this bundle has no test split).')
H.append('')
H.append('music_nd = share of her singing time (from beat 4) where Duet is consonant with her AND on a tone of the true chord AND not her note/octave;')
H.append('headline = mean of bach (17 val songs), folk (170), pop (300), pop909 (40).  double = share of time Duet sings her note or its octave.')
H.append('H = how many beats of her past melody the chord guesser may use.  Only the chord guess is limited; the chooser is unchanged and always')
H.append('remembers Duet\'s own previous note for voice leading (that is Duet\'s line, not her melody).')
H.append('  HMM: the filter is re-run over beats k-H .. k-1 plus the current beat k, starting from the song-start chord prior; H=0 = the current beat only')
H.append('       (her current note plus any earlier note inside the same beat).')
H.append('  GRU: each of the 3 GRUs is re-run from an empty state over the notes that started in the last H beats; H=0 = her current note only.')
H.append('  whole song = the unchanged systems (running HMM filter; GRU hidden state carried through the song).')
H.append('Files: m3_seq_v5h0/h4/h16 are the legacy wrappers that set DUET_HIST inside their own process (the outside environment was empty);')
H.append('rD_hist_m1_h*_v1 and rD_hist_mix_h*_v1 are new constant-only wrappers (filter.H = H; GRU ensemble built with hist = H). Scored with')
H.append('rescore_nd (runs_mini/v4_1101/nd/<name>/) and bench3 (one leaderboard row each).')
H.append('')
for title, ns in SYS:
    H.append(title)
    H.append(f"  {'window':26s} {'system':20s} headline  bach   folk   pop    pop909   double max")
    for w, n in zip(WIN, ns):
        r = ND_[n]
        H.append(f"  {w:26s} {n:20s} {r['headline_nd']:.4f}  " + ' '.join(f"{r[d]['music_nd']:.4f}" for d in DS)
                 + f"   {max(r[d]['double'] for d in DS):.3f}")
    a, z = ND_[ns[0]], ND_[ns[-1]]
    H.append(f"  history gain (whole song - current note): headline {z['headline_nd'] - a['headline_nd']:+.4f}; "
             + ', '.join(f"{d} {z[d]['music_nd'] - a[d]['music_nd']:+.4f}" for d in DS))
    H.append('')
m1, mix = ND_['m1_filter_v5lo'], ND_['r2_mix_v2']
H.append('What it shows:')
x16, x0 = LAST['rD_hist_m1_h16_v1']['rows'], LAST['m1_filter_v5lo']['rows']
same16 = ND_['rD_hist_m1_h16_v1'] == m1
diff16 = [(d, k) for d in DS for k in x16[d] if k != 'us_per_note' and x16[d][k] != x0[d][k]]
H.append(f"- HMM (m1_filter_v5lo): her past melody is worth only {m1['headline_nd'] - ND_['rD_hist_m1_h0_v1']['headline_nd']:+.4f}, almost all of it inside the last 4 beats."
         f" With 16 beats the scores equal the whole-song filter's")
H.append(f"  (music_nd and double {'identical to every digit' if same16 else 'NOT identical'}; bench3 rows identical except "
         + (', '.join(f"{d} {k} {x16[d][k]:.4f} vs {x0[d][k]:.4f}" for d, k in diff16) or 'nothing') + '):')
H.append('  the filter forgets notes older than about 16 beats on its own. pop gains nothing from history.')
H.append(f"- With no history the champion is no better than the HMM (H=0: {ND_['rD_hist_mix_h0_v1']['headline_nd']:.4f} vs {ND_['rD_hist_m1_h0_v1']['headline_nd']:.4f}). Its whole gain over m1_filter_v5lo"
         f" ({mix['headline_nd'] - m1['headline_nd']:+.4f})")
H.append(f"  comes from the GRUs remembering more of her melody (champion minus HMM at the same window: {ND_['rD_hist_mix_h0_v1']['headline_nd'] - ND_['rD_hist_m1_h0_v1']['headline_nd']:+.4f} at H=0,"
         f" {ND_['rD_hist_mix_h4_v1']['headline_nd'] - ND_['rD_hist_m1_h4_v1']['headline_nd']:+.4f} at 4 beats,"
         f" {ND_['rD_hist_mix_h16_v1']['headline_nd'] - ND_['rD_hist_m1_h16_v1']['headline_nd']:+.4f} at 16 beats, {mix['headline_nd'] - m1['headline_nd']:+.4f} for the whole song).")
H.append(f"- m3_seq_v5 gains {ND_['m3_seq_v5']['headline_nd'] - ND_['m3_seq_v5h0']['headline_nd']:+.4f} in music_nd, but on bach it LOSES with more history"
         f" ({ND_['m3_seq_v5h0']['bach']['music_nd']:.4f} -> {ND_['m3_seq_v5']['bach']['music_nd']:.4f}): the surer its chord guess, the more its own chooser")
H.append(f"  copies her (bach double {ND_['m3_seq_v5h0']['bach']['double']:.3f} -> {ND_['m3_seq_v5']['bach']['double']:.3f}), and music_nd counts copying as a miss.")
H.append('')
H.append('Old laptop numbers, m3_seq_v5, raw music, copying counted (bench3 headline, not music_nd): 0.658 / 0.674 / 0.695 / 0.710 / 0.720 for')
H.append('current note / last 1 beat / last 4 beats / last 16 beats / whole song.  The mini reproduces them (raw music, copying counted):')
H.append(f"{LAST['m3_seq_v5h0']['headline']:.4f} / (H=1 not re-run) / {LAST['m3_seq_v5h4']['headline']:.4f} / {LAST['m3_seq_v5h16']['headline']:.4f} / {LAST['m3_seq_v5']['headline']:.4f}."
         f" In music_nd the same ladder is {ND_['m3_seq_v5h0']['headline_nd']:.4f} / {ND_['m3_seq_v5h4']['headline_nd']:.4f} / {ND_['m3_seq_v5h16']['headline_nd']:.4f} / {ND_['m3_seq_v5']['headline_nd']:.4f}.")
H.append('')
H.append('repeat memory (prior from 4/8/16 bars ago) ≈ +0.004, not used')
H.append('')
H.append('val only, tuned on val.')
hist_txt = '\n'.join(H) + '\n'

print(board_txt)
print('=' * 100)
print(hist_txt)
if '--write' in sys.argv:
    for fn, txt in (('board.txt', board_txt), ('history.txt', hist_txt)):
        with open(os.path.join(REPORT, fn), 'x') as fh:
            fh.write(txt)
        print('wrote', os.path.join(REPORT, fn))
