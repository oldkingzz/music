"""Collect results/*.json into futures/results/d5.json. .venv/bin/python futures/d5_live_fx/make_results.py"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
OUT = HERE.parent / 'results' / 'd5.json'
REL = 'futures/d5_live_fx'

# assumed Linux USB-audio path (stated, not measured): ALSA 48 kHz, period 128 frames = 2.67 ms = one engine
# block (64 @ 24 kHz), 2 periods buffered each way, USB isochronous ~1 ms each way, ADC+DAC filters ~0.5 ms each
ALSA_IN_MS = 2 * 2.667
ALSA_OUT_MS = 2 * 2.667
USB_MS = 2 * 1.0
CODEC_MS = 2 * 0.5
F_LOW = 160.0                                       # lowest female f0 supported at full spec


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    lat, tim, sim, cl = load('latency.json'), load('timing_ecore.json'), load('sim.json'), load('clash.json')
    acc, caveats = [], []
    # ---------------- A5.1
    path_ms = max(d['delay_ms'] for d in lat['path_delay'])
    worst_alg = 1000 * (2.0 / 24000 + 1.0 / F_LOW)
    onset_ms = max(d['onset_ms'] for d in lat['onset'])
    m4 = [t for t in lat['timing_m4'] if t['sr'] == 24000][0]
    ec = [t for t in tim if t['sr'] == 24000][0] if tim else None
    ec48 = [t for t in tim if t['sr'] == 48000][0] if tim else None
    board_p99 = 2.0 * ec['p99_us'] / 1000 if ec else float('nan')
    board_mean = 2.0 * ec['mean_us'] / 1000 if ec else float('nan')
    fixed = ALSA_IN_MS + ALSA_OUT_MS + USB_MS + CODEC_MS
    e2e_typ = fixed + path_ms + board_p99
    e2e_worst = fixed + worst_alg + board_p99
    acc.append({'line': 'A5.1', 'text_zh': '麦克风进到喇叭出 ≤ 30 ms（端到端，板上实测）',
                'measured': (f'估计端到端 {e2e_typ:.1f} ms（最坏 {e2e_worst:.1f} ms，160 Hz 时 PSOLA 延迟 {worst_alg:.1f} ms）；'
                             f'仿真算法延迟：音频通路 {path_ms:.1f} ms、起音 {onset_ms:.1f} ms；每块（2.67 ms）计算 '
                             f'板上估计 {board_mean:.2f} ms 均值 / {board_p99:.2f} ms p99（M4 E 核 x2；M4 普通 '
                             f'{m4["mean_us"] / 1000:.2f}/{m4["p99_us"] / 1000:.2f} ms）；假设 ALSA 48 kHz 每周期 128 帧、'
                             f'每方向 2 个周期 + USB 2 ms + 编解码 1 ms = {fixed:.1f} ms'),
                'threshold': '<= 30 ms', 'status': 'needs_hardware',
                'evidence': f'{REL}/results/latency.json; {REL}/results/timing_ecore.json; {REL}/HARDWARE_TEST.md'})
    if ec and board_p99 > 2.667:
        caveats.append(f'板上计算 p99 估计 {board_p99:.2f} ms，超过一个块 2.67 ms：板上需要实测，可能要把块加大到 128@24 kHz（延迟 +5 ms 左右）')
    # ---------------- A5.2
    if sim:
        nominal = [r for r in sim if r['total_s'] == 180.0 and r['extra_db'] == 0.0]
        sweep = [r for r in sim if r['total_s'] == 60.0]

        def margin(rt, guard):
            pts = sorted([(0.0, any(r['howl']['howl'] for r in nominal if r['rt60'] == rt and r['guard'] == guard))] +
                         [(r['extra_db'], r['howl']['howl']) for r in sweep if r['rt60'] == rt and r['guard'] == guard])
            first = next((x for x, h in pts if h), None)
            stable = [x for x, h in pts if not h and (first is None or x < first)]
            return (max(stable) if stable else None), first

        txt, ok = [], True
        for rt in (0.3, 0.6):
            for g in (True, False):
                r = next(r for r in nominal if r['rt60'] == rt and r['guard'] == g)
                lo, hi = margin(rt, g)
                txt.append(f'RT60 {rt} s {"有防护" if g else "无防护"}：3 分钟{"啸叫" if r["howl"]["howl"] else "无啸叫"}，'
                           f'稳定裕量 ≥ {lo} dB' + (f'（{hi} dB 时啸叫）' if hi is not None else '（扫到 30 dB 仍不啸叫）'))
                if g and r['howl']['howl']:
                    ok = False
        acc.append({'line': 'A5.2', 'text_zh': '正常唱歌音量、麦克风和喇叭相距 1 m、一个房间里唱 3 分钟不啸叫',
                    'measured': '仿真：' + '；'.join(txt), 'threshold': '3 分钟无啸叫（房间实测）',
                    'status': 'needs_hardware', 'evidence': f'{REL}/results/sim.json'})
        # ---------------- A5.3
        gm = [r['gate_metrics'] for r in nominal if r['guard']]
        rel40 = max(g['release_to_-40dBFS_s']['max'] for g in gm)
        rel30p95 = max(g['release_to_sing-30dB_s']['p95'] for g in gm)
        rel30max = max(g['release_to_sing-30dB_s']['max'] for g in gm)
        fa = sum(g['false_trigger_events'] for g in gm)
        loud = sum(g['gap_output_above_-40dBFS_s'] for g in gm)
        gap = sum(g['gap_time_s'] for g in gm)
        n_off = sum(g['n_offsets'] for g in gm)
        st = 'pass' if (rel40 <= 0.2 and loud == 0) else 'fail'
        gol = [r.get('gate_metrics_open_loop') for r in nominal if r['guard'] and r.get('gate_metrics_open_loop')]
        fa_ol = sum(g['false_trigger_events'] for g in gol) if gol else None
        sb = [g.get('between_songs') for g in gm if g.get('between_songs')]
        sb_txt = ''
        if sb:
            sb_txt = (f'；两首歌之间的纯静音（只有房间噪声和回声）共 {sum(x["time_s"] for x in sb):.0f} s：门打开 '
                      f'{sum(x["gate_open_events"] for x in sb)} 次，喇叭高于 -40 dBFS {sum(x["output_above_-40dBFS_s"] for x in sb):.2f} s')
        acc.append({'line': 'A5.3', 'text_zh': '她停下后 0.2 s 内喇叭安静，且不重放房间噪声或自己的回声',
                    'measured': (f'仿真闭环（RT60 0.3/0.6 s，有防护，{n_off} 次收声）：降到 -40 dBFS 最长 {rel40:.3f} s；'
                                 f'更严口径降到演唱电平 -30 dB：p95 {rel30p95:.3f} s、最长 {rel30max:.3f} s；她乐句间的停顿共 {gap:.0f} s 内'
                                 f'门打开 {fa} 次（同一引擎无房间回路时 {fa_ol} 次，即由她自己的弱声/呼吸触发，不是回声），'
                                 f'喇叭高于 -40 dBFS 共 {loud:.2f} s' + sb_txt),
                    'threshold': '<= 0.2 s，无误触发', 'status': 'needs_hardware' if st == 'pass' else 'fail',
                    'evidence': f'{REL}/results/sim.json'})
        if rel30max > 0.2:
            caveats.append(f'按更严的"演唱电平 -30 dB"口径，少数收声（她慢慢渐弱时）超过 0.2 s（最长 {rel30max:.2f} s）')
    # ---------------- A5.4
    if cl:
        sy = cl['symbolic']
        lit = {ds: sy['literal'][ds] for ds in ('bach', 'folk', 'pop909', 'pop')}
        shf = {ds: sy['shifter'][ds] for ds in ('bach', 'folk', 'pop909', 'pop')}
        worst_lit = max(r['not_good'] for r in lit.values())
        au = cl['audio']
        a_all, a_snd, a_un = au['all'], au['all_sounding_interval'], au['all_unambiguous_notes']
        a_fem = au['female_only']
        sym_txt = '，'.join(f'{ds} {100 * r["not_good"]:.1f}%/{100 * r["harsh"]:.1f}%' for ds, r in lit.items())
        sh_txt = '，'.join(f'{ds} {100 * r["not_good"]:.1f}%' for ds, r in shf.items())
        st = 'pass' if worst_lit <= 0.10 and a_all['not_good'] <= 0.10 else 'fail'
        acc.append({'line': 'A5.4', 'text_zh': '她换音后的冲突时间 ≤ 她演唱时间的 10%（现有评分）',
                    'measured': (f'(a) 符号评测（她的音被听错 10%，延迟 {1000 * sy["lags_s"]["note"]:.0f} ms），不好/刺耳：{sym_txt}；'
                                 f'按移调器真实物理（和声 = 她当前音高下移三度）不好：{sh_txt}。'
                                 f'(b) 真实引擎跑 vocadito 40 段：不好 {100 * a_all["not_good"]:.1f}%（刺耳 {100 * a_all["harsh"]:.1f}%，'
                                 f'四度 {100 * a_all["fourth"]:.1f}%，静音 {100 * a_all["silent"]:.1f}%）；女声 {100 * a_fem["not_good"]:.1f}%；'
                                 f'按实际发声音程（不分别取整）{100 * a_snd["not_good"]:.1f}%；只算音高明确的音符 {100 * a_un["not_good"]:.1f}%；'
                                 f'去掉每个音头尾 60 ms {100 * au["all_note_cores"]["not_good"]:.1f}%。参照上限：把她自己的实际音高'
                                 f'零延迟、按正确音程下移，同样评分也有 {100 * au["ideal_follower_of_her_pitch"]["not_good"]:.1f}% 不好'
                                 f'（她的滑音、音头不准，标注是整音）'),
                    'threshold': '不好 <= 10%', 'status': st, 'evidence': f'{REL}/results/clash.json'})
    demos = []
    dd = HERE / 'demo'
    for f, lab in (('vocadito_37_harmony.mp3', 'vocadito 第 37 段：她 + Duet 实时下三度和声（40 段里冲突最少的一段，不代表平均）'),
                   ('vocadito_2_harmony.mp3', 'vocadito 第 2 段：她 + Duet 实时下三度和声（40 段里第 4 好）'),
                   ('room_3min_closed_loop.mp3', '3 分钟闭环房间仿真（有防护，RT60 0.6 s，正常音量）：麦克风听到的声音')):
        if (dd / f).exists():
            demos.append({'file': f'{REL}/demo/{f}', 'label_zh': lab, 'source': 'vocadito (Bittner et al.), CC BY 4.0; 仿真',
                          'page_safe': True})
    caveats += ['全部是仿真：房间是合成的扩散场脉冲响应（固定不变），真实房间里人走动会让回声路径变化，回声消除器只在开场 2 s 试音时学习一次',
                '延迟里的 USB/ALSA 部分是假设值，不是测量值；需要 Vincent 按 HARDWARE_TEST.md 做回环实测',
                '音频评测把她的标注音和 Duet 的音高分别取整；业余歌手常常唱在两个半音之间（约 10-25% 的时间），这部分取整噪声算进了"不好"',
                'A5.4 符号评测里"和声 = 下三度(听到的音)"会把听错的音也算错；而真实移调器是把她当前的实际音高下移三度，所以听错几乎不造成冲突（0.2-0.3%）',
                '板上计算时间是 M4 E 核 x2 的估计；E 核测量时本机还有其他任务在跑，p99 偏高（另一次更忙时 p99 估计 4.8 ms）',
                '没有 test 划分：A5.4 符号评测用 bench 的 val 歌曲，引擎参数只在少量 vocadito 段上看过，没有留出测试集']
    s = {'id': 'd5', 'title_zh': '实时和声效果器',
         'updated': subprocess.run(['date', '-Iseconds'], capture_output=True, text=True).stdout.strip(),
         'summary_zh': '', 'acceptance': acc, 'demos': demos, 'local_only': [],
         'how_to_run': '.venv/bin/python futures/d5_live_fx/run_all.py', 'caveats_zh': caveats}
    stat = {a['line']: a['status'] for a in acc}
    s['summary_zh'] = ('能用的：2.67 ms 一块的实时引擎（YIN、调内下三度、流式 TD-PSOLA、门、回声消除、限幅）在仿真里算法延迟约 2 ms，'
                       '估计端到端约 20 ms；仿真房间里 3 分钟不啸叫，有防护时稳定裕量 ≥ 24 dB（无防护只有 6 dB），停唱后 0.07 s 内降到 -40 dBFS。'
                       '不行的：在真实业余歌声（vocadito）上冲突时间 20-25%，没过 10% 线（符号评测 6-7% 过线）；延迟和啸叫只是仿真，要板子和房间实测。')
    OUT.write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
