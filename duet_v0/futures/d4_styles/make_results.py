"""Assemble futures/results/d4.json from out/board_{val,test}.json, out/styles.json, out/timing.json."""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
rel = lambda p: os.path.relpath(p, REPO)  # noqa: E731


def style_id_proxy(st):
    """Leave-one-melody-out nearest-centroid style classifier on the arrangement statistics."""
    keys = ['onsets_per_s', 'swung_offbeat_share', 'straight_eighth_share', 'seventh_chord_share',
            'substitution_share', 'voices_max', 'chords_per_beat', 'mean_note_beats']
    X, y, mel = [], [], []
    for style, rows in st['stats'].items():
        for i, r in enumerate(rows):
            X.append([r[k] for k in keys])
            y.append(style)
            mel.append(i)
    X = np.array(X, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ok = 0
    for i in range(len(X)):
        tr = [j for j in range(len(X)) if mel[j] != mel[i]]
        cents = {s: X[[j for j in tr if y[j] == s]].mean(0) for s in set(y)}
        pred = min(cents, key=lambda s: np.sum((X[i] - cents[s]) ** 2))
        ok += pred == y[i]
    return ok / len(X), len(X)


def rob_text(st):
    r = st.get('robustness_30_test_melodies')
    if not r:
        return ''
    return (f"Robustness on {r['bach']['harsh']['n']} more test melodies (Bach/folk/POP909, 32 beats): min Bach harsh "
            f"{r['bach']['harsh']['min']:.3f}, pop avoid {r['pop']['avoid']['min']:.3f}, jazz avoid {r['jazz']['avoid']['min']:.3f}; "
            f"mean pop harsh {r['pop']['harsh']['mean']:.3f}, jazz harsh {r['jazz']['harsh']['mean']:.3f}.")


def main():
    bv = json.load(open(os.path.join(OUT, 'board_val.json')))
    tp = os.path.join(OUT, 'board_test.json')
    bt = json.load(open(tp)) if os.path.exists(tp) else None
    st = json.load(open(os.path.join(OUT, 'styles.json')))
    tm = json.load(open(os.path.join(OUT, 'timing.json')))
    B = bt or bv
    split = 'test' if bt else 'val (test not yet run)'
    ours = B['tuned']['chosen_on_val']
    base2 = B['tuned']['base2_on_val']
    g1 = B['gain'][f'{ours}_vs_hmm_causal']
    g2 = B['gain'][f'{ours}_vs_{base2}']
    acc = B['acc']
    per = ', '.join(f"{ds} {acc[ours][ds]:.3f} vs {acc['hmm_causal'][ds]:.3f}/{acc[base2][ds]:.3f}" for ds in ('bach', 'folk', 'pop', 'pop909'))
    clear = g1['mean']['gain'] >= 0.10 and g1['mean']['ci95'][0] > 0
    a43 = {'line': 'A4.3', 'text_zh': '从旋律预测原曲和弦（每拍 7 个调内三和弦），明显超过现有 HMM（增益 >= 10 个百分点且 95% 置信区间不含 0）',
           'measured': (f"{split}: mean accuracy {acc[ours]['mean']:.3f} ({ours}) vs current causal HMM {acc['hmm_causal']['mean']:.3f} "
                        f"(gain {100 * g1['mean']['gain']:+.1f} pts, 95% CI [{100 * g1['mean']['ci95'][0]:+.1f}, {100 * g1['mean']['ci95'][1]:+.1f}]); "
                        f"vs offline HMM ({base2}) {acc[base2]['mean']:.3f} (gain {100 * g2['mean']['gain']:+.1f} pts, CI "
                        f"[{100 * g2['mean']['ci95'][0]:+.1f}, {100 * g2['mean']['ci95'][1]:+.1f}]). Per dataset ours vs causal/offline: {per}. "
                        f"Board estimate {2 * tm['ecore']['chord_model_us_per_beat']:.0f} us/beat ({tm['m4']['chord_model_us_per_beat']:.0f} us/beat M4)."),
           'threshold': 'gain over current HMM >= 10 points, CI excludes 0',
           'status': 'pass' if clear else 'fail',
           'evidence': rel(tp if bt else os.path.join(OUT, 'board_val.json'))}
    sm = st['summary']
    mins = {'bach': sm['bach']['harsh']['min'], 'pop': sm['pop']['avoid']['min'], 'jazz': sm['jazz']['avoid']['min']}
    ok42 = all(v >= 0.90 for v in mins.values())
    bo = st.get('bach_own_reference', {}).get('no_clash_harsh', {}).get('all')
    a42 = {'line': 'A4.2', 'text_zh': '三种风格各自与她的不冲突比例 >= 90%（巴赫用刺耳音程版；流行/爵士用避免音版，另报刺耳版）',
           'measured': (f"min over 4 melodies (time-weighted mean): Bach harsh {mins['bach']:.3f} ({sm['bach']['harsh']['mean']:.3f}); "
                        f"pop avoid {mins['pop']:.3f} ({sm['pop']['avoid']['mean']:.3f}), pop harsh {sm['pop']['harsh']['min']:.3f} "
                        f"({sm['pop']['harsh']['mean']:.3f}); jazz avoid {mins['jazz']:.3f} ({sm['jazz']['avoid']['mean']:.3f}), "
                        f"jazz harsh {sm['jazz']['harsh']['min']:.3f} ({sm['jazz']['harsh']['mean']:.3f}). "
                        + (f"Bach's own A/T/B on the same chorale excerpt (harsh): {bo:.3f}. " if bo is not None else '')
                        + rob_text(st)),
           'threshold': '>= 0.90 per style (Bach: harsh rule; pop/jazz: avoid-note rule)',
           'status': 'pass' if ok42 else 'fail', 'evidence': rel(os.path.join(OUT, 'styles.json'))}
    acc_id, n_id = style_id_proxy(st)
    s_b, s_p, s_j = (sm[s]['stats_mean'] for s in ('bach', 'pop', 'jazz'))
    a41 = {'line': 'A4.1', 'text_zh': '盲听：听者说对风格 >= 80%（需要真人；已备盲听包，并给自动代理指标）',
           'measured': (f"needs listeners; blind kit 4 melodies x 3 styles in futures/d4_styles/blind/. Proxy: 7th-chord share "
                        f"Bach {s_b['seventh_chord_share']:.2f} / pop {s_p['seventh_chord_share']:.2f} / jazz {s_j['seventh_chord_share']:.2f}; "
                        f"substitutions jazz {s_j['substitution_share']:.2f} (others 0); swung off-beat onsets jazz {s_j['swung_offbeat_share']:.2f} "
                        f"vs pop {s_p['swung_offbeat_share']:.2f}; straight 8th onsets pop {s_p['straight_eighth_share']:.2f}; "
                        f"onsets/s Bach {s_b['onsets_per_s']:.1f} pop {s_p['onsets_per_s']:.1f} jazz {s_j['onsets_per_s']:.1f}; "
                        f"chord changes per beat Bach {s_b['chords_per_beat']:.2f} pop {s_p['chords_per_beat']:.2f} jazz {s_j['chords_per_beat']:.2f}; "
                        f"leave-one-melody-out nearest-centroid style ID on these 8 statistics {acc_id:.2f} ({n_id} clips)."),
           'threshold': '>= 80% correct style naming by listeners', 'status': 'needs_human',
           'evidence': rel(os.path.join(HERE, 'blind', 'INSTRUCTIONS.md'))}
    demos = [{'file': d['file'], 'label_zh': f"同一旋律依次配成 巴赫 / 流行 / 爵士（{d['melody']}）", 'source': d['source'] + '; ' + d['licence'],
              'page_safe': True} for d in st['demos']]
    res = {'id': 'd4', 'title_zh': '换个风格', 'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'summary_zh': (f"三种风格都能自动配出来，与她不冲突比例（巴赫按刺耳音程、流行/爵士按避免音）最低 {min(mins.values()):.1%}；"
                          f"和弦预测在 4 个数据集上平均 {acc[ours]['mean']:.1%}，比现有实时 HMM 高 {100 * g1['mean']['gain']:.0f} 个百分点（{split.split(' ')[0]}）。"
                          f"风格能否被人听出来还没有真人盲听；流行/爵士若按刺耳音程版算，最低只有 {min(sm['pop']['harsh']['min'], sm['jazz']['harsh']['min']):.0%}。"),
           'acceptance': [a41, a42, a43], 'demos': demos, 'local_only': [],
           'how_to_run': 'bash futures/d4_styles/run_all.sh',
           'caveats_zh': [
               '和弦标签只有 7 个调内三和弦，离调和弦被折算成最接近的三和弦；流行数据（Hooktheory）只用于训练和打分，从未渲染成音频。',
               '基线 1 是实时系统的因果判断（提前 L+delta 决定），本任务是离线看整首旋律，所以增益里有一大块来自“能看后文”；对离线 HMM 的增益更能说明模型本身。',
               '流行/爵士的“刺耳版”不冲突比例低：三和弦/七和弦下任何经过音都会和某个和弦音成二度，这是定义使然，验收用避免音版。',
               '巴赫风格用 D1 的 futures/common/satb.py（硬性禁止刺耳音程），所以刺耳版是 100%，比巴赫本人在同一段上的值还高；这说明它很保守，不代表更像巴赫。',
               'vocadito 片段的拍子是从音符起点估出来的（estimate_beat），真人唱得不完全按拍，流行/爵士伴奏和她可能有轻微错拍；音符用的是人工标注而不是自动转写。',
               'A4.2 的和声是从她的真实音符（乐谱或人工标注）算出来的，没有经过音高识别这一步；换成麦克风实录要再测。巴赫本人的参考值用的是女高音起音切片，内声部在一个旋律音内部的移动被忽略。',
               '风格可辨认度只有统计代理（七和弦比例、替代和弦、摇摆八分音符），合成音色很简单，真人盲听还没做。'],
           'board_estimate': {'chord_model_us_per_beat': 2 * tm['ecore']['chord_model_us_per_beat'],
                              'arrange_3_styles_s_per_melody': 2 * tm['ecore']['arrange_s_per_melody_3_styles'],
                              'note': 'board_estimate = 2 x M4 efficiency-core time (estimate)', 'm4': tm['m4']}}
    weak = [d for d in ('bach', 'folk', 'pop', 'pop909') if g2[d]['ci95'][0] <= 0]
    if weak:
        res['caveats_zh'].insert(1, f"相对离线 HMM，{'、'.join(weak)} 上的增益置信区间包含 0（"
                                 + '；'.join(f"{d} {100 * g2[d]['gain']:+.1f} 点" for d in weak)
                                 + "），即在这些数据上新模型没有明显好过把旧 HMM 改成离线解码。")
    os.makedirs(os.path.join(REPO, 'futures', 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, 'futures', 'results', 'd4.json'), 'w'), indent=1, ensure_ascii=False)
    write_report(res, B, split, ours, base2, st, tm, acc_id)
    print(json.dumps([(a['line'], a['status'], a['measured'][:200]) for a in res['acceptance']], ensure_ascii=False, indent=1))


def write_report(res, B, split, ours, base2, st, tm, acc_id):
    acc = B['acc']
    systems = ['hmm_causal', 'hmm_causal_benchproto', 'hmm_fb', 'hmm_vit', 'style_hmm', 'bigru', 'bigru_vit', 'ens']
    L = ['# D4 换个风格 (style switch): report', '',
         'Regenerate everything: `bash futures/d4_styles/run_all.sh` (the one-time test run was '
         '`.venv/bin/python futures/d4_styles/chord_board.py --test`; it refuses to run twice).', '',
         '## A4.3 chord-prediction leaderboard (' + split + ')', '',
         'Task: the whole clean melody of a song -> the diatonic triad (7 classes) of every beat; scored on every labelled beat. '
         'Songs: ' + ', '.join(f"{k} {v}" for k, v in B['songs'].items()) + '.', '',
         '| system | bach | folk | pop | pop909 | mean |', '|---|---|---|---|---|---|']
    for sname in systems:
        if sname in acc:
            a = acc[sname]
            L.append(f"| {sname}{' (chosen on val)' if sname == ours else ''} | " + ' | '.join(f"{a[d]:.3f}" for d in ('bach', 'folk', 'pop', 'pop909', 'mean')) + ' |')
    L += ['', '- hmm_causal: baseline 1, the PDR ChordHMM (Bach train) run through duet_beat/planner.run exactly as the realtime '
          'system (decides 72 ms before each beat), on the clean melody. hmm_causal_benchproto: the same under the bench protocol '
          '(10 % misheard notes, from beat 8); its val Bach value reproduces the leaderboard_v2 "chord" number (0.319) exactly.',
          f'- hmm_fb / hmm_vit: baseline 2, the same ChordHMM decoded offline (forward-backward / Viterbi); {base2} was better on val.',
          '- style_hmm: one HMM per style, duration-weighted pitch-class emissions + onset weight, separate downbeat emissions, '
          'positional transitions; kappa / onset weight / lambda / decoder tuned per style on val.',
          '- bigru: 2-layer bidirectional GRU (96 units/direction) on 39 per-beat features (pc durations, note onsets, rest, bar '
          'position, meter, style id), trained on MPS on the union of the 4 train splits (Bach x12, folk x3, pop909 x2 oversampled), '
          'epoch chosen on val; inference in numpy (max |logit| difference to torch ~1e-6, asserted).',
          '- bigru_vit / ens: BiGRU log-posteriors + style-HMM transitions (and, for ens, w x style-HMM emissions), Viterbi; w, beta on val.', '',
          'Gains of the chosen system (95 % bootstrap CI over songs, paired, 2000 resamples; mean = mean of the 4 dataset gains):', '']
    for k, g in B['gain'].items():
        L.append(f"- {k}: " + '; '.join(f"{d} {100 * g[d]['gain']:+.1f} [{100 * g[d]['ci95'][0]:+.1f}, {100 * g[d]['ci95'][1]:+.1f}]" for d in ('bach', 'folk', 'pop', 'pop909', 'mean')))
    L += ['', f"Compute: {tm['m4']['chord_model_us_per_beat']:.0f} us/beat on M4; board estimate {2 * tm['ecore']['chord_model_us_per_beat']:.0f} us/beat "
          '(2 x M4 E-core, estimate) -> a 255-beat song in ' + f"{2 * tm['ecore']['chord_model_ms_per_255beat_song']:.0f} ms.", '',
          '## A4.2 no-clash per style (rendered note events vs her true notes)', '',
          '| melody | Bach harsh | pop avoid | pop harsh | jazz avoid | jazz harsh |', '|---|---|---|---|---|---|']
    for n, row in st['melodies'].items():
        L.append(f"| {n} | {row['bach']['no_clash_harsh']:.3f} | {row['pop']['no_clash_avoid']:.3f} | {row['pop']['no_clash_harsh']:.3f} | "
                 f"{row['jazz']['no_clash_avoid']:.3f} | {row['jazz']['no_clash_harsh']:.3f} |")
    bo = st.get('bach_own_reference')
    if bo:
        L += ['', f"Bach's own A/T/B against the soprano on the same chorale excerpt ({bo['chorale']}, harsh rule): {bo['no_clash_harsh']['all']:.3f} "
              f"(per voice {', '.join(f'{k} {v:.3f}' for k, v in bo['no_clash_harsh']['per_voice'].items())}). Per-voice values for every style are in out/styles.json."]
    if st.get('robustness_30_test_melodies'):
        L += ['', rob_text(st)]
    L += ['', '- Bach: D1 futures/common/satb.py (chord Viterbi + voicing DP with a near-hard ban on harsh intervals against her), A/T/B sung by the synthetic singer.',
          '- pop: per half bar the triad maximizing the pop-style BiGRU log-posterior minus avoid-note and harsh time against her; split into two one-beat '
          'chords when clearly better. Block chord on the half-bar + broken-chord eighths (piano), bass root on beats 1 and 3, soft straight-eighth shaker.',
          '- jazz: the same base harmony as diatonic 7ths; before each downbeat chord change a tritone substitution, ii-V, or secondary dominant is '
          'tried (rotating) and kept only if its avoid-note time is no worse than the plain chord and <= 10 % of her time there. Rootless 4-note '
          'voicings (3-7 with 9/13/5, the variant with least clash), Charleston-type swung comping, walking bass (root, chord tones, chromatic approach '
          'chosen to avoid clashes), swung ride.', '',
          '## A4.1 style identity (needs listeners)', '',
          'Blind kit: futures/d4_styles/blind/ (12 clips = 4 melodies x 3 styles, random ids, KEY.json, INSTRUCTIONS.md in Chinese). Automatic proxy (mean over 4 melodies):', '',
          '| style | 7th-chord share | substitution share | swung off-beat onsets | straight 8th onsets | onsets/s | max voices | chord changes/beat | mean note (beats) |', '|---|---|---|---|---|---|---|---|---|']
    for sname in ('bach', 'pop', 'jazz'):
        m = st['summary'][sname]['stats_mean']
        L.append(f"| {sname} | {m['seventh_chord_share']:.2f} | {m['substitution_share']:.2f} | {m['swung_offbeat_share']:.2f} | {m['straight_eighth_share']:.2f} | {m['onsets_per_s']:.2f} | {m['voices_max']:.1f} | {m['chords_per_beat']:.2f} | {m['mean_note_beats']:.2f} |")
    L += ['', f'Leave-one-melody-out nearest-centroid classifier on these 8 statistics names the style of {acc_id:.0%} of the 12 clips '
          '(a weak proxy: it shows the arrangements differ in the intended ways, not that listeners hear it).', '',
          '## Demos (page-safe, futures/d4_styles/demo/, sources in demo/SOURCES.md)', '']
    for d in st['demos']:
        L.append(f"- {d['file']} ({d['bytes'] / 1e6:.2f} MB): {d['source']}; {d['licence']}")
    L += ['', '## What failed / caveats / next', ''] + ['- ' + c for c in res['caveats_zh']] + [
        '- Next: real listeners for A4.1; a key/beat tracker for real singing instead of annotation notes; richer timbres; '
        'chords beyond the 7 diatonic triads in the leaderboard labels.']
    open(os.path.join(HERE, 'REPORT.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
