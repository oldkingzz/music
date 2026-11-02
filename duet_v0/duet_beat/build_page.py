"""Build the results page (one HTML file plus media) from out/results.json and out/audio_results.json.

Run:  .venv/bin/python duet_beat/build_page.py <target_dir>
Copies the demo audio (mp3) into <target_dir>/media and writes <target_dir>/index.html.
Term pronunciations (<target_dir>/audio/*.mp3) are made with macOS `say` if missing.
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

DEMOS = [  # (demo name, heading, what to listen for)
    ('arp_slow_vec', '你的琶音 · 流行和弦节奏 · 12 格比例',
     '第二遍开始唱（第 9 拍起）。和弦没换的拍{ok_other}；换和弦的那一拍，{n_chg} 次里猜错 {chg_wrong} 次。'),
    ('arp_bach_long', '你的琶音 · 巴赫设定 · 只看最长的音',
     '一拍三个音只看到一个，又按巴赫的习惯以为和弦多半每拍都换，没把握就不出声，所以一大半时间沉默。'),
    ('twinkle_bach_long', '小星星 · 巴赫设定',
     '和弦只猜对一半，但 Duet 大多停在 G 上：G 同时属于 I 和 V，所以它的音大多仍在和弦里。'),
    ('twinkle_slow_long', '小星星 · 流行和弦节奏',
     '和弦没换的拍大多猜对；换和弦的那一拍，{n_chg} 次只猜对 {chg_right} 次，在她唱出新和弦的音之前，模型只能凭转移概率猜换不换。Duet 的 {wrong_all} 个错音里，{wrong_chg} 个出在这些拍上。'),
    ('twinkle_slow_vibrato', '小星星 · 换成带颤音和滑音的歌手',
     '和上一条结果完全一样：耳朵听得准，瓶颈不在耳朵。'),
    ('bach_typical', '巴赫众赞歌 BWV 324（测试集里按规则选的典型一首）',
     '开头十几拍几乎每拍换和弦，中间有两段各 8 拍不换。Duet 的和弦大多猜错，但它在 D（同时属于 I、iii、V）上连唱了 25 拍，所以唱的音有六成多仍在和弦里。'),
    ('twinkle_drift', '小星星 · 她比节拍器快 1.5%',
     '她越唱越超前。Duet 跟唱几拍后，发现她的起音离拍点超过 80 毫秒，就停唱（对拍监测）。'),
]
SYSTEM_ORDER = ['hmm_long_N4', 'hmm_vec_N4', 'hmm_long_N1', 'B0_tonic', 'B1_prior', 'B2_rule', 'B3_peek', 'B4_oracle']
SYSTEM_ZH = {
    'hmm_long_N4': '主系统：隐马尔可夫，最长的音，规划 4 拍',
    'hmm_vec_N4': '同上，观测换成 12 格比例',
    'hmm_long_N1': '同上，只规划 1 拍',
    'B0_tonic': 'B0 永远猜 I 和弦',
    'B1_prior': 'B1 只用转移、从不听',
    'B2_rule': 'B2 仓库旧规则：她上一个音往下三度或六度',
    'B3_peek': 'B3 偷看这一拍（不守因果，上限）',
    'B4_oracle': 'B4 直接告诉它巴赫的和弦（上限）',
}
TERMS = ['hidden Markov model', 'baseline', 'harmonic rhythm', 'vibrato', 'click track', 'common tone',
         'receding horizon', 'causality', 'front end']
SPOKEN = {}
VOICE = 'Reed (英语（美国）)'


def slug(t):
    return re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')


def say(t):
    return f'<span class="say" data-say="{html.escape(t)}" role="button" tabindex="0" title="点击听发音">{html.escape(t)}</span>'


def make_audio(target):
    os.makedirs(os.path.join(target, 'audio'), exist_ok=True)
    for t in TERMS:
        p = os.path.join(target, 'audio', slug(t) + '.mp3')
        if os.path.exists(p):
            continue
        aiff = p[:-4] + '.aiff'
        subprocess.run(['say', '-v', VOICE, '-r', '150', '-o', aiff, SPOKEN.get(t, t)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', aiff, '-ac', '1', '-b:a', '64k', p], check=True)
        os.remove(aiff)                      # the temporary aiff this function just wrote


CT = {0: {0, 4, 7}, 1: {2, 5, 9}, 2: {4, 7, 11}, 3: {0, 5, 9}, 4: {2, 7, 11}, 5: {0, 4, 9}, 6: {2, 5, 11}}


def demo_facts(d):
    """Counts quoted in the demo notes, computed from the run itself (beats from J on)."""
    b, J, t = d['beats'], d['J'], d['tonic']
    chg = [k for k in range(max(J, 1), len(b)) if b[k]['z_true'] != b[k - 1]['z_true']]
    oth = [k for k in range(J, len(b)) if k not in chg]
    wrong = lambda ks: sum(1 for k in ks if b[k]['h'] >= 0 and (b[k]['h'] - t) % 12 not in CT[b[k]['z_true']])  # noqa: E731
    right = lambda ks: sum(1 for k in ks if b[k]['z_hat'] == b[k]['z_true'])  # noqa: E731
    return {'n_chg': len(chg), 'chg_right': right(chg), 'chg_wrong': len(chg) - right(chg),
            'ok_other': '全部猜对' if right(oth) == len(oth) else f'猜对 {right(oth)} / {len(oth)}',
            'wrong_all': wrong(range(J, len(b))), 'wrong_chg': wrong(chg)}


def stats_line(d, m):
    tail = f'<span class="muted">{d["bpm"]:g} BPM · {d["K"]} 拍 · 第 {d["J"] + 1} 拍起才可能唱</span>'
    if d['name'] == 'twinkle_drift':
        sung = [b['k'] for b in d['beats'] if b['action'] in ('sing', 'hold')]
        stop = next(b['k'] for b in d['beats'] if b['action'] == 'stopped')
        return (f'<span>她 {d["her_bpm"]:g} BPM，节拍器 {d["bpm"]:g} BPM</span><span>第 <b>{sung[0] + 1}</b> 到 <b>{sung[-1] + 1}</b> 拍跟唱</span>'
                f'<span>第 <b>{stop + 1}</b> 拍起停唱</span>{tail}')
    return (f'<span><b>{pct(m["m1"])}</b> 和弦猜对</span><span><b>{pct(m["m2"])}</b> Duet 的音在和弦里</span>'
            f'<span><b>{pct(m["m5"])}</b> 出声的拍</span>{tail}')


def pct(x):
    return '—' if x != x else f'{100 * x:.1f}%'


def main(target):
    res = json.load(open(os.path.join(OUT, 'results.json')))
    aud = json.load(open(os.path.join(OUT, 'audio_results.json')))
    R, diffs = res['results'], res['diffs']
    demos = {d['name']: d for d in aud['demos']}
    os.makedirs(os.path.join(target, 'media'), exist_ok=True)
    for name, _, _ in DEMOS:
        shutil.copy2(os.path.join(OUT, 'media', f'{name}.mp3'), os.path.join(target, 'media', f'{name}.mp3'))
    make_audio(target)

    main_r, b0, b3, b2 = R['hmm_long_N4'], R['B0_tonic'], R['B3_peek'], R['B2_rule']
    arp = demos['arp_slow_vec']['metrics']
    fe = aud['front_end']
    D = lambda k: diffs[k]  # noqa: E731
    fmtd = lambda k: f"{100 * D(k)[0]:+.1f} 个百分点（区间 {100 * D(k)[1][0]:+.1f} 到 {100 * D(k)[1][1]:+.1f}）"  # noqa: E731
    fmtp = lambda k: f"{100 * D(k)[0]:.1f} 个百分点（区间 {100 * D(k)[1][0]:.1f} 到 {100 * D(k)[1][1]:.1f}）"  # noqa: E731

    # ---------------------------------------------------------------- demo cards
    cards = []
    payload = {}
    for name, head, note in DEMOS:
        d = demos[name]
        m = d['metrics']
        note = note.format(**demo_facts(d))
        payload[name] = {k: d[k] for k in ('K', 'M', 'J', 'tonic', 'sop', 'beats', 'bpm')}
        cards.append(f'''
      <article class="demo" id="{name}">
        <h3>{html.escape(head)}</h3>
        <p class="note">{html.escape(note)}</p>
        <audio controls preload="none" src="media/{name}.mp3"></audio>
        <div class="stats">{stats_line(d, m)}</div>
        <div class="roll" data-demo="{name}" tabindex="0" role="img" aria-label="{html.escape(head)}：她的旋律、Duet 的音和每拍的和弦。左右方向键逐拍查看。"></div>
        <details><summary>逐拍表</summary><div class="tablebox"><table class="beats" data-demo="{name}"></table></div></details>
      </article>''')

    # ---------------------------------------------------------------- the big table
    rows = []
    for key in SYSTEM_ORDER:
        r = R[key]
        cls = ' class="main"' if key == 'hmm_long_N4' else ''
        m4s = "—" if r["m4"] != r["m4"] else f"{r['m4']:.1f}"
        ci = r['ci']
        rows.append(f'<tr{cls}><td>{html.escape(SYSTEM_ZH[key])}</td>'
                    f'<td class="n">{pct(r["m1"])}</td><td class="n">{pct(r["m2"])}</td><td class="n">{pct(r["m2b"])}</td>'
                    f'<td class="n">{pct(r["m3"])}</td><td class="n">{m4s}</td>'
                    f'<td class="n">{pct(r["m5"])}</td><td class="n">{r["u"]:+.2f}</td></tr>')
    dots = [{'key': k, 'label': SYSTEM_ZH[k], 'm1': R[k]['m1'], 'm2': None if k == 'B1_prior' else R[k]['m2'],
             'ci1': R[k]['ci']['m1'], 'ci2': R[k]['ci']['m2']} for k in SYSTEM_ORDER]

    A4 = res['learned']['A4']
    stay_bach = [A4[p][i][i] for p in range(4) for i in range(7)]
    stay_pos = [sum(A4[p][i][i] for i in range(7)) / 7 for p in range(4)]
    slow = aud['slow_on_bach']
    curve = res['tuning']['q_curve_val']
    qrows = ''.join(f'<tr><td class="n">{c["q_min"]:.1f}</td><td class="n">{pct(c["m2"])}</td><td class="n">{pct(c["m5"])}</td>'
                    f'<td class="n">{c["u"]:+.3f}</td></tr>' for c in curve)
    data_info = res['data']

    page = TEMPLATE
    rep = {
        '__CARDS__': '\n'.join(cards),
        '__ROWS__': '\n'.join(rows),
        '__QROWS__': qrows,
        '__PAYLOAD__': json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        '__DOTS__': json.dumps(dots, ensure_ascii=False),
        '__ARP_M1__': pct(arp['m1']), '__ARP_M2__': pct(arp['m2']),
        '__MAIN_M1__': pct(main_r['m1']), '__B0_M1__': pct(b0['m1']), '__B3_M1__': pct(b3['m1']),
        '__MAIN_M2__': pct(main_r['m2']), '__B0_M2__': pct(b0['m2']), '__B2_M2__': pct(b2['m2']), '__B3_M2__': pct(b3['m2']),
        '__D_M1_B0__': fmtp('hmm_long_N4-B0_tonic:m1'), '__D_M2_B0__': fmtp('hmm_long_N4-B0_tonic:m2'),
        '__D_M2_B2__': fmtp('hmm_long_N4-B2_rule:m2'), '__D_M1_B3__': fmtp('B3_peek-hmm_long_N4:m1'),
        '__B3GAIN__': f"{100 * D('B3_peek-hmm_long_N4:m1')[0]:.1f}",
        '__FE_MIS__': str(fe['clean']['mismatch']), '__FE_MIS_MULTI__': str(fe['clean']['mismatch_multi']),
        '__D_N__': fmtd('hmm_long_N4-hmm_long_N1:m2'), '__D_VEC__': fmtd('hmm_vec_N4-hmm_long_N4:m1'),
        '__STAY0__': f'{100 * stay_pos[0]:.0f}%', '__STAY1__': f'{100 * stay_pos[1]:.0f}%',
        '__STAY2__': f'{100 * stay_pos[2]:.0f}%', '__STAY3__': f'{100 * stay_pos[3]:.0f}%',
        '__SLOW_M1__': pct(slow['m1']), '__SLOW_M2__': pct(slow['m2']),
        '__FE_C_FULL__': pct(fe['clean']['agree_full']), '__FE_C_PART__': pct(fe['clean']['agree_partial']),
        '__FE_V_FULL__': pct(fe['vibrato']['agree_full']), '__FE_V_PART__': pct(fe['vibrato']['agree_partial']),
        '__FE_C_M1__': pct(fe['clean']['audio']['m1']), '__FE_V_M1__': pct(fe['vibrato']['audio']['m1']),
        '__FE_S_M1__': pct(fe['clean']['score']['m1']), '__FE_C_M2__': pct(fe['clean']['audio']['m2']),
        '__FE_V_M2__': pct(fe['vibrato']['audio']['m2']), '__FE_S_M2__': pct(fe['clean']['score']['m2']),
        '__FE_N__': f"{fe['clean']['n_beats']:,}",
        '__N_TRAIN__': str(data_info['train']['chorales']), '__N_VAL__': str(data_info['val']['chorales']),
        '__N_TEST__': str(data_info['test']['chorales']), '__B_TEST__': f"{data_info['test']['beats']:,}",
        '__B_TRAIN__': f"{data_info['train']['beats']:,}",
        '__QMIN__': f"{res['tuning']['chosen']['main']['q_min']:.1f}", '__GAMMA__': f"{res['tuning']['chosen']['main']['gamma']:g}",
        '__M4__': f"{main_r['m4']:.1f}", '__M4_ORACLE__': f"{R['B4_oracle']['m4']:.1f}",
        '__FE_C_P95__': f"{fe['clean']['drift_p95_abs_ms']:.0f}", '__FE_V_P95__': f"{fe['vibrato']['drift_p95_abs_ms']:.0f}",
    }
    for p, s_mix in enumerate((0.3, 0.9, 0.6, 0.9)):        # songs.slow_harmony: bar, weak, half bar, weak
        rep[f'__SSTAY{p}__'] = f'{100 * (s_mix + (1 - s_mix) * stay_pos[p]):.0f}%'
    for t in TERMS:
        rep[f'__T:{t}__'] = say(t)
    for k, v in rep.items():
        page = page.replace(k, v)
    left = re.findall(r'__[A-Z0-9_]+__|__T:[^_]+__', page)
    assert not left, left
    with open(os.path.join(target, 'index.html'), 'w') as fh:
        fh.write(page)
    print('wrote', os.path.join(target, 'index.html'), len(page), 'bytes')


TEMPLATE = open(os.path.join(HERE, 'page_template.html')).read() if os.path.exists(os.path.join(HERE, 'page_template.html')) else ''

if __name__ == '__main__':
    main(sys.argv[1])
