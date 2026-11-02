<!-- Phase D 草稿（report writer）。src 路径约定：RUN/ = /Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/ ；V3/ = /Users/san/claude_sandbox/music/duet_v0/futures/v3_nolatency/ ；SI/ = /Users/san/claude_sandbox/music/duet_v0/futures/v4_selfiter/ ；DER1、DER2 = RUN/report_drafts/tools/derived_numbers_v1.json、derived_numbers_v2.json（只把结果文件里的数相减或求平均；脚本 derive_numbers*.py 在同一目录）。 -->
# 早安 Vincent：Duet v4 自迭代结果（run v4_1101，11-01）

- **可上台（presentable）冠军 `r2_mix_v2`：0.6444**，旧模型 `m1_filter_v5lo` 0.6206（+0.0238）；换随机种子（seed 2）复核：0.6470 对 0.6208。<!-- src: RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json; RUN/nd/m1_filter_v5lo/results/nd_m1_filter_v5lo.json; RUN/seed2/r2_mix_v2.json; RUN/seed2/m1_filter_v5lo.json; 差值 DER1 diff.r2_mix_v2-m1 -->
- **最高分 `r3_style_mix_v3`：0.6532，但不可上台**：它的风格感知 HMM（style-aware HMM：通用、pop、pop909 三份 HMM 按她的节奏加权）按 §4.3 严格读法不算"那个 HMM"；算不算，由你定。<!-- src: RUN/nd/r3_style_mix_v3/results/nd_r3_style_mix_v3.json; RUN/state_R4.json champion.note; V3/systems/r2_style_hmm_v1.py -->
- 改了什么：实时猜和弦改成对数线性混合（log-linear mix）$b \propto b_{\mathrm{HMM}}^{0.3}\, b_{\mathrm{GRU}}^{0.7}$，七成听 3 个 GRU 的平均、三成听 m1 的 HMM；选音器（chooser）和它的 6 个权重不变。<!-- src: V3/systems/r2_mix_v2.py (ALPHA = 0.7); RUN/state_R2.json champion.what -->

<!-- src（下表）：music_nd、double = RUN/nd/<name>/results/nd_<name>.json；括号内差值 = DER1 diff（r2_mix_v2-m1、r3_style_mix_v3-m1）；seed 2 = RUN/seed2/<name>.json；val 歌数 = RUN/report/board.txt；double 取 4 个数据集的最大值 = DER1 double_max。 -->
| 公平分 music_nd（val 歌数） | m1（旧） | r2_mix_v2（可上台） | r3_style_mix_v3（最高分） |
|---|---:|---:|---:|
| bach（17） | 0.6655 | 0.6853（+0.0198） | 0.7028（+0.0373） |
| folk（170） | 0.6766 | 0.7126（+0.0359） | 0.7178（+0.0412） |
| pop（300） | 0.5601 | 0.5908（+0.0307） | 0.5963（+0.0363） |
| pop909（40） | 0.5804 | 0.5891（+0.0087） | 0.5959（+0.0156） |
| **平均（headline）** | **0.6206** | **0.6444（+0.0238）** | **0.6532（+0.0326）** |
| seed 2 平均 | 0.6208 | 0.6470 | 0.6527 |
| 唱她同音/八度（double，最大） | 0.048 | 0.055 | 0.043 |

music_nd = 她唱歌的时间里（第 4 拍起），Duet "和她协和 + 是真和弦的音 + 不是她的同音/八度"的比例。参照：三度下方（third_below）0.5054；告诉它真和弦（m4_oracle）0.8388。<!-- src: V3/rescore_nd.py; V3/bench3.py (J = 4); RUN/nd/third_below/results/nd_third_below.json; V3/results/nd_m4_oracle.json -->

**点子（idea）与结论**
- E1 GRU 猜和弦 + m1 选音（0.6440）：folk 大跳率（big）0.0212 超上限 0.0195；第 2 次 w_big=0.6 过关（0.6433）但不如 E2 → 结束。<!-- src: DER2 ideas.E1 (headline_nd, big_folk), keep_rule_folk_big_limit_vs_m1; RUN/state_R2.json ideas E1 -->
- E2 HMM×GRU 混合 $\alpha=0.7$（`r2_mix_v2`）→ **保留，可上台冠军**；$\alpha$ = 0 / 0.5 / 0.7 / 0.85 / 1 → 0.6206 / 0.6428 / 0.6444 / 0.6454 / 0.6440，0.85 的 folk 大跳 0.0205 超限。<!-- src: DER2 ideas.E1 / ideas.E2 (headline_nd, big_folk); RUN/state_R2.json ideas E2 trend -->
- E3 拍子时长先验（prior）：原样无收益（0.6139 / 0.6212）；改成风格感知 HMM（0.6296）过关但低于 E2 → 用进 E2×E3。<!-- src: DER1 m1_tweaks_rt_music_nd (why_e DUR_pooled_lam1.5 / lam1.0); DER2 ideas.E3 r2_style_hmm_v3; RUN/state_R2.json ideas E3 -->
- E2×E3 风格感知 HMM×GRU $\alpha=0.5$（`r3_style_mix_v3`）→ **保留，最高分，不可上台**。<!-- src: DER2 ideas.E2xE3; RUN/state_R3.json -->
- E4 按她的音会盖住的几拍算契合：R3 最好 +0.0027，R4 −0.0003 → **败两次，死（dead）**。<!-- src: DER2 ideas.E4_R3 (r3_cover_fit_v3), ideas.E4_R4; RUN/state_R4.json ideas_round_4 E4 -->
- E5 按 10% 听错算与她真实音的协和：R3 −0.0008 ~ +0.0005，R4 −0.0005 / −0.0042 → **败两次，死**。<!-- src: DER2 ideas.E5_R3, ideas.E5_R4; RUN/state_R4.json ideas_round_4 E5 -->
- 第二波（E6 重训 GRU、E7 束搜索）没开：它只在第一波没有可上台冠军时才开。<!-- src: SI/PROGRAM.md §4.4; RUN/state_R4.json loop.stop_reason -->

* **警报（alerts）：** 无（心跳 START 全有 END、返回码全 0；校验和 10/10；磁盘空闲 103.9 GB）。<!-- src: RUN/state_R4.json alerts, checks_after_round_4; RUN/hb/ 里没有 ALERTS_*.md -->
* **盲听：** 没做二选一。你的原话：「好吧问题不是这个，几乎所有的音乐都是好听的，这些都是合格的。关键的问题其实发生在duet，即那个配和弦的地方（实时）。所有的问题都发生在那里，所以你需要帮我解决那个的问题。其他静态阶段都是没问题的。」→ 4 轮全在修实时猜和弦，原因见 `DIAGNOSIS_zh.md`。<!-- src: SI/listen/result.json vincent_said -->
* **音频 `report/audio/`：** 这 3 段 30 秒片段里 m1 反而比两个冠军都高（如 Twinkle 0.694 对 0.583）；进步是平均值（17 首 bach 整首赢 9 平 1 输 7，片段赢 6 平 1 输 10），短片段不一定听得出。<!-- src: RUN/report/audio/index.md; RUN/phaseD/A/clip_context.json summary (champion = r2_mix_v2) -->
* **一共打分：** 28 个候选（R1 2、R2 11、R3 12、R4 3）；另有约 40 + 27 + 15 个只诊断、不算候选的配置，和 9 个历史长度消融（history ablation）。<!-- src: RUN/state_R4.json variants_scored_in_loop, variants_note; DER2 scored_candidates_counted_here; RUN/report/board.txt -->
* **停止原因：** §4.4 的点子用完了（E4、E5 各败两次），第二波没触发；循环 14:45–16:09 共 4 轮，没碰到任何上限。<!-- src: RUN/state_R4.json loop -->
* **全部数字是 val 上的，也在 val 上调（没有 test）。**

**你接下来：** 读 `report/CARD_new_zh.md`，再在聊天里做测验（quiz）：≥ 8/10 且 5 个数字（0.505、0.621、0.644、0.839、double 的意思）都在 ±0.01 内，`r2_mix_v2` 才是周二模型；否则讲 m1（`CARD_m1_zh.md`）。<!-- src: SI/PROGRAM.md §5 -->

**只有你能决定删不删（我什么都没删）：**
- `/Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/nd/third_below/runs_mini/`：Phase A 并行 cd 事故留下的两个嵌套重复文件 `.../nd/m1_filter_v5lo/results/nd_m1_filter_v5lo.json`、`.../nd/m3_seq_v5/results/nd_m3_seq_v5.json`（数字与正式文件相同）。<!-- src: RUN/state_R0.json incidents -->
- `/Users/san/claude_sandbox/music/duet_v0/futures/v4_selfiter/listen/ab.html`、`key.json`、`audio/`：第一版听力页（v1），从没给你看过（你听的是 `listen/v2/`）。<!-- src: RUN/state_R0.json interpretations, phase_b -->
