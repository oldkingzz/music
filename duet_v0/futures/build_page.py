"""Rebuild team_status/duet-futures.html from futures/results/*.json and futures/verify/*.json.

Each card keeps the approved acceptance lines and shows, per line, the measured number, the builder's
status and the verifier's verdict. Page-safe demo mp3s are copied to team_status/futures_audio/.
Writes team_status/files_futures.json (published path -> source) for the Artifact publish.
"""
import html
import json
import os
import re
import shutil

F = os.path.dirname(os.path.abspath(__file__))
DUET = os.path.dirname(F)
TS = os.path.join(DUET, '..', 'team_status')
PAGE = os.path.join(TS, 'duet-futures.html')
AUD = os.path.join(TS, 'futures_audio')

CARDS = [
    ('d1', '录一遍，配一遍', '她唱一句，几秒后扬声器放出“她 + Duet 配的三个声部”，听起来是一段四部合唱。',
     '按一下开始，清唱 4–8 小节，停下。', '她的原声，下面加上 Duet 合成的中音、次中音、低音，一起放出来。',
     [('A1.1', '唱完到开始播放不超过 5 秒，全部在板子上算。'),
      ('A1.2', '配出的声部和她的音不冲突的时间 ≥ 95%，且没有平行五度、平行八度。'),
      ('A1.3', '盲听：把 Duet 配的和 Bach 真实配的混在一起，听的人分不出来的比例接近一半。')],
     '板子上的一键 demo + 3 段示范录音'),
    ('d2', '她自己的合唱团', '和声不用合成声音，而是把她自己的录音变调成三个声部，听起来是三个“她”在给她和声。',
     '唱一句，或者一首歌的一段。', '四个她：原唱在上面，三个变调的她在下面跟着唱，音色、呼吸、咬字都是她的。',
     [('A2.1', '变调最多 7 个半音时，听的人觉得“是同一个人”，不像机器声。'),
      ('A2.2', '三个声部用的音和“录一遍，配一遍”同一套，好听的标准一样。'),
      ('A2.3', '一段 30 秒的录音在板子上 30 秒内处理完。')],
     '板子上的 demo + 原声 / 合唱对比录音'),
    ('d3', '第二遍就会了', '第一遍 Duet 只听；第二遍她再唱，Duet 已经知道她下一拍要唱什么，现场实时和她一起唱。',
     '同一首歌唱两遍。第二遍可以快一点、慢一点，也可以中途停顿。', '第二遍里，Duet 跟着她的速度走，在拍点上就唱对，不用等她先开口。',
     [('A3.1', '第二遍好听的时间 ≥ 95%（现在的实时方案是 79%）。'),
      ('A3.2', '她快 20% 或慢 20%，Duet 都能跟上，不脱拍。'),
      ('A3.3', '她中途停 2 秒再接着唱，Duet 在一拍内重新对上。')],
     '现场 demo 的保底方案 + 一段两遍对比录音'),
    ('d4', '换个风格', '同一句旋律，Duet 配成 Bach、流行、爵士三种和声，听一遍就能听出区别。',
     '唱一句，然后选风格。', '同一句旋律依次放三遍：众赞歌式的四部和声、流行歌式的和弦伴唱、带七和弦的爵士和声。',
     [('A4.1', '盲听：随机放一种，听的人能说对是哪个风格的比例 ≥ 80%。'),
      ('A4.2', '每种风格都过同一条底线：和她的音不冲突的时间 ≥ 90%。'),
      ('A4.3', '和弦预测榜上，猜原作和弦的准确率明显高于现在的 HMM。')],
     '三风格对比录音 + 和弦预测榜成绩'),
    ('d5', '实时和声效果器', '她一开口，扬声器里就同时多出一个她：她的声音被实时移到调内下三度，和她一起唱，像舞台上的和声效果器。',
     '先选好调，然后对着麦克风随便唱，不用事先录。', '她自己的声音低三度跟着她走；她换音，和声几乎同时换，听不出先后。',
     [('A5.1', '延迟：从她的声音进麦克风到和声从扬声器出来，端到端 ≤ 30 ms，在板子上实测。'),
      ('A5.2', '不啸叫：扬声器开到正常演唱音量，麦克风和扬声器在同一个房间相距 1 米，连续唱 3 分钟不出现啸叫或回授。'),
      ('A5.3', '她停下不唱时，扬声器 0.2 秒内静音，不把房间噪声或自己的回声再放出来。'),
      ('A5.4', '调内音换音后的冲突时间，按现有打分不超过她唱歌时间的 10%。')],
     '板子上的实时 demo + 延迟实测记录 + 3 分钟不啸叫录像'),
]
STATUS = {'pass': ('通过', 'ok'), 'fail': ('未通过', 'bad'), 'needs_human': ('要人听', 'wait'),
          'needs_hardware': ('要上板实测', 'wait'), None: ('未完成', 'none')}
VERDICT = {'confirmed': '复核确认', 'corrected': '复核修正', 'refuted': '复核推翻', 'unverifiable': '复核无法确认'}


def esc(s):
    return html.escape(str(s))


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def norm_status(s):
    s = (s or '').strip().lower()
    return s if s in STATUS else None


# (deliverable, line) -> (status, short Chinese reason): the lead's call after reading the verdicts
OVERRIDES = {
    ('d2', 'A2.2'): ('fail', '复核发现输出音频上有 31 处平行（多是 Duet 声部之间同度重复）。修复后重排声部：平行降到 0 处，但按音符取整的不冲突 94.5%，差一点到 95%；代价是常常只有两个声部在唱。修复后未复核。'),
    ('d3', 'A3.2'): ('needs_hardware', '复核：只在合成歌手上、而且要开非默认的 --acc 开关才过；把真人分轨拉慢 20% 的代理测试里 Duet 明显掉拍（79.6% 降到 55.7%），所以改成要真人和板子实测。'),
    ('d3', 'A3.3'): ('needs_hardware', '迭代后仿真 5/5 在她重新开口的第一个音就对上（靠听到她换气提前 0.08 秒准备），未复核；能不能听到真人的换气、板子的输出延迟，都要实测。'),
}
MEASURED_ZH = {
    ('d1', 'A1.1'): '板子估算 2.4 秒（复核在高负载下测得 4.1 秒），线是 5 秒；还没上板。',
    ('d1', 'A1.2'): '40 段真人录音端到端：不冲突 96.0%，平行五/八度 0 处；巴赫测试集 100%、0 处。修复后复核确认，但修复规则参考过部分测试曲目。',
    ('d1', 'A1.3'): '盲听包 10 对已备好。Duet 和巴赫用同一个和弦只占 33.5%，声部也更少跳进，多半听得出来。',
    ('d2', 'A2.1'): '盲听包已备好。移调后共振峰只偏 28/38 Hz（普通变调 98/388 Hz），频谱包络差 3 dB（普通变调 9 dB）。',
    ('d2', 'A2.2'): '修复后，输出音频上平行 0 处（原 31 处）；不冲突按音符取整 94.5%（原 91.7%），另一种取整 97.6%；和第 1 条同一套和弦：计划层 100%，输出 82.6%。',
    ('d2', 'A2.3'): '30 秒录音板子估算 10.1 秒（复核 8.8 秒），线是 30 秒；还没上板。',
    ('d3', 'A3.1'): '合成歌手第二遍 96.9%；真人合唱分轨只有 51.2%（只用第一遍的音频排练）。搭建 agent 原报 56.6%，那一版偷用了第一遍的人工标注。',
    ('d3', 'A3.2'): '合成歌手快 20% / 慢 20%：96.2% / 95.2%，没脱拍（要开 --acc 开关）；把真人分轨拉慢 20% 时掉到 55.7%。',
    ('d3', 'A3.3'): '合成歌手 5 个停顿：迭代后 5/5 在她重新开口的第一个音就对上（原来 4/5，最慢一次 2 拍）。',
    ('d4', 'A4.1'): '盲听包 12 段已备好；一个按 8 个统计量分类的小程序能认对 10/12（83%）。',
    ('d4', 'A4.2'): '最低一首：巴赫 100%，流行 95.0%，爵士 95.5%；改用扒谱得到的音重算仍过（95.3% / 98.6% / 97.4%）。',
    ('d4', 'A4.3'): '测试集准确率 0.590，现在的 HMM 0.308（高 28.2 个点，95% 区间 27.0–29.5）；比离线 HMM 也高 14.8 个点。',
    ('d5', 'A5.1'): '仿真估计端到端 18.8 ms（最坏 22.8 ms），其中 13.7 ms 是假设的 USB 声卡缓冲；MacBook 自带声卡测不了这条（见下方实机测试）。',
    ('d5', 'A5.2'): '仿真房间 3 分钟不啸叫，有防护时稳定余量 ≥ 24 dB（无防护只有 6 dB）。',
    ('d5', 'A5.3'): '仿真里停唱后最慢 0.115 秒降到 −40 dBFS，纯静音 0 次误触发；MacBook 真实房间 40 秒底噪里也是 0 次误触发。',
    ('d5', 'A5.4'): '真人录音不好听 21.7%（迭代前 24.5%），线是 10%；理想的零延迟跟随按同样打分也有约 14%。',
}
DEMO_PICK = {
    'd2': ['v11_original', 'v11_choir', 'v11_voices_only', 'v7_original', 'v7_choir', 'v12_choir'],
    'd3': ['greensleeves_two_takes', 'greensleeves_faster_x1.2', 'greensleeves_slower_x0.8',
           'greensleeves_pause_2s_v2', 'greensleeves_pause_2s_before_fix'],
}


def card(cid, title, one, did, heard, lines, deliv, res, ver, files, fix=None, fver=None):
    acc = {a.get('line', '').strip().upper(): a for a in (res or {}).get('acceptance', [])}
    facc = {a.get('line', '').strip().upper(): a for a in (fix or {}).get('acceptance', [])}
    vl = {v.get('line', '').strip().upper(): v for v in (ver or {}).get('lines', [])}
    fvl = {v.get('line', '').strip().upper(): v for v in (fver or {}).get('lines', [])}
    rows = []
    for lid, text in lines:
        a, v = acc.get(lid, {}), vl.get(lid, {})
        fixed = lid in facc and facc[lid].get('measured') != a.get('measured')
        if fixed:
            a, v = facc[lid], fvl.get(lid, {})
        st = norm_status(a.get('status'))
        if v.get('verified_status') == 'refuted' and st == 'pass':
            st = 'fail'
        reason = ''
        if (cid, lid) in OVERRIDES:
            st, reason = OVERRIDES[(cid, lid)]
        label, cls = STATUS[st]
        measured = a.get('measured', '')
        if v.get('verified_status') == 'corrected' and v.get('verified_measured'):
            measured = f"{v['verified_measured']}（原报 {measured}）"
        vtxt = ('修复后' + VERDICT.get(v.get('verified_status'), '，未复核')) if fixed else VERDICT.get(v.get('verified_status'), '未复核' if res else '')
        note = v.get('note', '') if v.get('verified_status') in ('corrected', 'refuted') else ''
        note = reason or note
        short = MEASURED_ZH.get((cid, lid))
        detail = f'<details><summary>原始数字</summary><p>{esc(measured)}</p></details>' if short and measured else ''
        rows.append(
            f'<li><span class="chip {cls}">{label}</span><div class="ln"><p class="t">{esc(text)}</p>'
            f'<p class="m">{esc(short or measured or "—")}'
            f'{f" · <span class=v>{esc(vtxt)}</span>" if vtxt else ""}</p>'
            f'{f"<p class=n>{esc(note[:260])}</p>" if note else ""}{detail}</div></li>')
    demos = []
    pool, seen = [], set()
    for d in (fix or {}).get('demos', []) + (res or {}).get('demos', []):
        b = os.path.basename(d.get('file', ''))
        if b and b not in seen:
            seen.add(b)
            pool.append(d)
    if fix:
        olds = {os.path.basename(d.get('file', '')).replace('_v2', '') for d in fix.get('demos', []) if '_v2' in d.get('file', '')}
        pool = [d for d in pool if os.path.basename(d.get('file', '')) not in olds]
    if cid in DEMO_PICK:
        base = lambda d: os.path.splitext(os.path.basename(d['file']))[0].replace('_v2', '')  # noqa: E731
        order = {k.replace('_v2', ''): i for i, k in enumerate(DEMO_PICK[cid])}
        pool = sorted([d for d in pool if base(d) in order], key=lambda d: order[base(d)])
    for d in pool:
        if len(demos) >= 6:
            break
        src = os.path.join(DUET, d.get('file', '').replace('project/duet_v0/', '')) if d.get('file') else ''
        src = src if os.path.exists(src) else os.path.join(DUET, '..', '..', d.get('file', ''))
        if not d.get('page_safe') or not os.path.exists(src) or not src.endswith('.mp3'):
            continue
        if os.path.getsize(src) > 1_600_000 or re.search(r'(dcs|ecd|esmuc|hooktheory)', src, re.I):
            continue
        rel = f'futures_audio/{cid}/{os.path.basename(src)}'
        os.makedirs(os.path.join(AUD, cid), exist_ok=True)
        shutil.copy2(src, os.path.join(TS, rel))
        files[rel] = os.path.join(TS, rel)
        demos.append(f'<label><span>{esc(d.get("label_zh") or os.path.basename(src))}</span>'
                     f'<audio controls preload="none" src="{rel}"></audio></label>')
    summary = (fix or res or {}).get('summary_zh', '')
    caveats = (fix or res or {}).get('caveats_zh', [])[:3]
    ver_sum = (ver or {}).get('summary_zh', '')
    return f'''
  <article class="card" id="{cid}">
    <h2>{esc(title)}</h2>
    <p class="one">{esc(one)}</p>
    <div class="flow">
      <div class="in"><span class="lab">她做什么</span>{esc(did)}</div>
      <div class="arrow" aria-hidden="true">→</div>
      <div class="out"><span class="lab">她听到什么</span>{esc(heard)}</div>
    </div>
    <div class="done"><span class="lab">验收结果</span><ul class="acc">{''.join(rows)}</ul></div>
    {f'<p class="sum">{esc(summary)}</p>' if summary else '<p class="sum">还没有结果。</p>'}
    {f'<p class="vsum">复核：{esc(ver_sum)}</p>' if ver_sum else ''}
    {('<ul class="cav">' + ''.join(f'<li>{esc(c)}</li>' for c in caveats) + '</ul>') if caveats else ''}
    {f'<div class="players">{"".join(demos)}</div>' if demos else ''}
    <p class="meta"><span>交付：<b>{esc(deliv)}</b></span></p>
  </article>'''


def main():
    files = {}
    cards, tally = [], {'ok': 0, 'bad': 0, 'wait': 0, 'none': 0}
    for c in CARDS:
        res = load(os.path.join(F, 'results', f'{c[0]}.json'))
        ver = load(os.path.join(F, 'verify', f'{c[0]}.json'))
        fix = load(os.path.join(F, 'results', f'{c[0]}_fix.json'))
        fver = load(os.path.join(F, 'verify', f'{c[0]}_fix.json'))
        cards.append(card(*c, res, ver, files, fix, fver))
    for m in re.finditer(r'class="chip (\w+)"', ''.join(cards)):
        tally[m.group(1)] += 1
    page = open(PAGE).read()
    bak = PAGE + '.bak-1030'
    if not os.path.exists(bak):
        shutil.copy2(PAGE, bak)
    page = open(bak).read()                       # always rebuild from the approved page, not from a previous build
    head = page.split('<article class="card"')[0]
    head = head.replace('<title>Duet 五个玩法</title>', '<title>Duet 五个玩法</title>')
    lede_new = (f'<p class="tally"><span class="chip ok">通过 {tally["ok"]}</span>'
                f'<span class="chip wait">要人听或上板 {tally["wait"]}</span>'
                f'<span class="chip bad">未通过 {tally["bad"]}</span>'
                + (f'<span class="chip none">未完成 {tally["none"]}</span>' if tally['none'] else '') + '</p>')
    head = re.sub(r'<p class="tally">.*?</p>', '', head, flags=re.S)
    head = head.replace('  </header>', '    ' + lede_new + '\n  </header>', 1)
    css = '''
.chip { display: inline-block; font: 600 12.5px/1 var(--body); padding: 5px 8px; border-radius: 999px; white-space: nowrap; }
.chip.ok { background: var(--duet); color: var(--sheet); }
.chip.bad { background: var(--verm); color: var(--sheet); }
.chip.wait { background: var(--orange); color: var(--sheet); }
.chip.none { background: var(--rule); color: var(--ink2); }
.tally { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 0; }
ul.acc { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
ul.acc li { display: grid; grid-template-columns: 92px 1fr; gap: 10px; align-items: start; }
ul.acc .chip { justify-self: start; }
.ln { min-width: 0; display: grid; gap: 2px; }
.ln p { margin: 0; }
.ln .m { font: 500 14px/1.5 var(--body); color: var(--ink); overflow-wrap: anywhere; }
.ln .m .v { color: var(--ink2); font-family: var(--body); }
.ln .n { font-size: 13.5px; color: var(--ink2); }\n.ln details { font-size: 13px; color: var(--ink2); }\n.ln details summary { cursor: pointer; }\n.ln details p { margin: 4px 0 0; overflow-wrap: anywhere; }\n.hwt { background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; padding: 18px 22px; display: grid; gap: 8px; }\n.hwt h2 { font: 700 20px/1.3 var(--display); margin: 0; }\n.hwt ul { margin: 0; padding-left: 1.2em; display: grid; gap: 4px; }
.sum { margin: 0; }
.vsum { margin: 0; font-size: 14.5px; color: var(--ink2); }
.cav { margin: 0; padding-left: 1.2em; font-size: 14.5px; color: var(--ink2); display: grid; gap: 2px; }
.players { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 8px 16px; }
.players label { display: grid; gap: 4px; font-size: 13.5px; color: var(--ink2); min-width: 0; }
.players audio { width: 100%; height: 34px; }
@media (max-width: 560px) { ul.acc li { grid-template-columns: 1fr; } }
'''
    if '.chip {' not in head:
        head = head.replace('footer { color:', css + 'footer { color:', 1)
    hwt = ('''
  <section class="hwt" id="mac">
    <h2>MacBook 实机测试（10/30 11:45–11:52）</h2>
    <ul>
      <li>MacBook 自带的麦克风和扬声器是两个独立设备。缓冲开到最小时，系统报的延迟是输入 75 ms + 输出 45 ms；连续测往返，结果在 100 到 800 ms 之间乱跳。所以 30 ms 那条线在 MacBook 上测不了，要用板子上同一块 USB 声卡测。</li>
      <li>效果器在 MacBook 上真的实时跑起来了：每 2.67 ms 一块，计算平均 0.17 ms，p99 0.23–0.29 ms（M4，机器当时还很忙）。</li>
      <li>扬声器放歌声，自带麦克风只收到 −32 dBFS，房间底噪是 −43 dBFS（100 Hz 以下最强）。效果器的门一直没打开，所以 Duet 没出声，也没有啸叫或误触发。要测真效果，需要外接麦克风，或者人对着麦克风唱。</li>
    </ul>
  </section>''')
    foot = ('\n  <footer>只写最终交付和验收结果。“要人听”的项附了盲听包，“要上板实测”的项是仿真数字，'
            '板子到手后按各自的 HARDWARE_TEST.md 实测。真人示范音频来自 vocadito（Bittner et al.），CC BY 4.0；'
            'Hooktheory 的歌只用于训练和打分，不出现在任何录音里。</footer>\n</div>\n')
    out = head + ''.join(cards) + hwt + foot
    open(PAGE, 'w').write(out)
    json.dump(files, open(os.path.join(TS, 'files_futures.json'), 'w'), indent=1)
    print('ok', tally, len(files), 'audio files')


if __name__ == '__main__':
    main()
