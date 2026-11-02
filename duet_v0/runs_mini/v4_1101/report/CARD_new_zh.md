<!-- CARD_new_zh.md 定稿（Phase D finalizer，2026-11-01）：底稿是 report_drafts/CARD_new_zh.md（v1），加上事实核查与规格核查的全部更正；没用精简稿 CARD_new_zh_v2.md，因为它丢了术语表里的例子，而两稿要改的地方一样。src 注释里的路径都相对于 /Users/san/claude_sandbox/music/duet_v0/ 。 -->
# Duet 新模型卡片：r2_mix_v2

> 全部是 val 上的数字，也在 val 上调（这个包没有 test）。

- **身份：** 可上台的冠军（presentable champion）。Vincent 通过问答（10 题对 ≥ 8 题，且 5 个数都在 ±0.01 内）它才是周二 11-03 的模型，否则周二讲老模型 m1_filter_v5lo。
- **val** = 验证集（validation split）：bach（巴赫众赞歌, Bach chorales）17 首、folk（民歌, folk songs）170 首、pop（流行歌, pop songs）300 首、pop909（另一套流行歌）40 首，共 527 首、46310 个音；**test** = 测试集（test split），这个包故意没有。
- **公平分 music_nd**（fair score：不算抄她的和声得分）：跳过开头 4 拍（beat），按她真唱的时间加权，Duet 的音同时满足三条的时间比例——和她的真音协和（consonant）、是这一拍真和弦（chord）的和弦音（chord tone）、不是她的同音或八度（unison / octave；唱这两种叫抄她, doubling）；不唱也算失败。**总分（headline）** = 4 个数据集（dataset）各自 music_nd 的平均。

$$\text{music\_nd}=\frac{\sum_i\sum_{k\ge 4}\mathrm{ov}(i,k)\;\mathbb{1}\big[\,|m_i-h_i| \bmod 12\in\{3,4,7,8,9\}\ \wedge\ h_i\in c_k\,\big]}{\sum_i\sum_{k\ge 4}\mathrm{ov}(i,k)}$$

$m_i$：她第 $i$ 个音的真音高；$h_i$：Duet 的音；$c_k$：第 $k$ 拍的真和弦（只算有标签的拍）；$\mathrm{ov}(i,k)$：第 $i$ 个音和第 $k$ 拍重叠多久。

- **要背的 5 个数：** third_below 0.505、m1 0.621、r2_mix_v2 0.644、告诉真和弦时的天花板（ceiling）0.839（m4_oracle；用 m1 自己的选音规则是 0.840）、抄她 = 唱她的同音或八度。

<!-- src: runs_mini/v4_1101/state_R4.json presentable_champion.tuesday_model; futures/v4_selfiter/PROGRAM.md §1.3 (no test split) and §5 (quiz: >= 8/10, 5 numbers within +-0.01); runs_mini/v4_1101/report/board.txt (val songs 17/170/300/40; music_nd and double definitions); runs_mini/v4_1101/r2/A/check_copy_20261101_151621.json reproduction_v1_w_big_0.4.all_val_vs_r1 (527 songs, 46310 notes); futures/v3_nolatency/rescore_nd.py (iv in CONSONANT, iv != 0, chord tone; silence in the denominator; headline_nd = sum of the 4 dataset music_nd / 4); futures/v3_nolatency/bench3.py (J = 4); duet_beat/theory.py (CONSONANT = {0,3,4,7,8,9}); runs_mini/v4_1101/nd/third_below/results/nd_third_below.json 0.5054; runs_mini/v4_1101/nd/m1_filter_v5lo/results/nd_m1_filter_v5lo.json 0.6206; runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json 0.6444; futures/v3_nolatency/results/nd_m4_oracle.json 0.8388; runs_mini/v4_1101/diag/diag_rt_v1.json headline_mean_of_4.music_nd.told 0.8403 -->

## 1. 一句话

r2_mix_v2 = 老模型 m1_filter_v5lo 的选音器（chooser）一点不改，只把"实时猜和弦（real-time chord guess）"换成隐马尔可夫模型（HMM, hidden Markov model）和 3 个门控循环单元（GRU, gated recurrent unit）的对数线性混合（log-linear mix）：$b\propto b_{HMM}^{0.3}\,b_{GRU}^{0.7}$。

<!-- src: futures/v3_nolatency/systems/r2_mix_v2.py (ALPHA = 0.7; Mix.belief); runs_mini/v4_1101/state_R2.json champion.what, champion.presentable_why -->

## 2. 三句话讲清楚

她每唱出一个音，Duet 在同一刻（零延迟, zero latency）猜出现在是 7 个和弦里的哪一个，再在她下方挑一个音，一直唱到她的下一个音。猜和弦时把两个猜测相乘：HMM 根据她刚唱过的音和"和弦通常怎么走"稳稳地跟（占 0.3），3 个 GRU 读她到目前为止唱过的整段旋律（占 0.7），两边都看好的和弦才会胜出。挑音的规则和 m1 完全一样，只因为和弦猜得更准，top-1 猜对率（top-1 accuracy）从 0.434 升到 0.491，music_nd 就从 0.6206 升到 0.6444，4 个数据集都涨。

<!-- src: runs_mini/v4_1101/diag/diag_champs_v1.json table_4dp.*.headline.acc (m1_filter_v5lo 0.4335, r2_mix_v2 0.4913); runs_mini/v4_1101/nd/m1_filter_v5lo/results/nd_m1_filter_v5lo.json; runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json -->

## 3. 输入和输出

**她唱第 $i$ 个音时，Duet 拿到的全部信息：**
- 这个音的起音时刻（onset），以拍为单位；
- 这个音**听到的**音高，用 MIDI 音高（MIDI note number）表示：每个音独立地有 10 % 的概率被听错（mishearing）成另外 11 个音级（pitch class）之一，取最近的八度；每首歌怎么听错由固定的随机种子（random seed）决定；
- 上一个音的长度（这个音会唱多久还不知道）；
- 调的主音（tonic），所有和弦都按大调（major key）表示；拍号（meter，每小节几拍）；拍子网格（beat grid，每一拍是小节（bar）里的第几拍）；
- **没有**和弦标签（chord label），**没有**任何未来的信息。

**Duet 回答：** 一个女中音音域（alto range，MIDI 53–77，即 F3–F5）里、严格低于她听到的音的音，从她起音那一刻唱到她下一个音开始；她下方找不到协和的音就不唱。val 的 46310 个音里有 291 个不唱（约 0.6 %）。

**零延迟：** 基准测试（benchmark）bench3 假定 Duet 在她起音的同一刻就拿到音高并给出答案，所以只比"唱得好不好"。计算本身每个音约 0.15 毫秒（bench3 实测 152–155 µs，Mac mini M4 的 CPU；m1 约 12–13 µs）。

<!-- src: futures/v3_nolatency/bench3.py (docstring; run_song passes onset, heard midi, prev_dur, tonic/M/pos/K; no chord labels; one note held until her next onset); duet_beat/bench.py heard() (NOISE = 0.10, one of the other 11 pitch classes, nearest octave, seed crc32(id) ^ 5160); duet_beat/theory.py (ALTO = (53, 77) "F3..F5"; major keys only); futures/v3_nolatency/systems/m1_filter_v5lo.py step (below = 1: h <= min(77, midi - 1); returns -1 if no candidate); runs_mini/v4_1101/review/r2_mix_v2.json d_range (sung 46019, silent 291); futures/v3_nolatency/leaderboard_v3.jsonl last val rows: r2_mix_v2 us_per_note 152.5-154.5, m1_filter_v5lo 11.9-13.5 -->

**每个音内部 4 步（和代码顺序一致）：**
1. 3 个 GRU 先各读入这个音的 24 个特征（feature），更新各自的隐藏状态（hidden state），各输出 7 个和弦的概率，取平均得 $b_{GRU}$。
2. HMM 结算已经结束的拍：用她在那一拍唱过的音（按时长加权）更新 7 个和弦的概率，再按转移表（transition table）推到下一拍。（第 1、2 步互不依赖，顺序不影响结果。）
3. HMM 把她这个音也算作当前拍的证据，得到 $b_{HMM}$；两者相乘再归一化（normalize），得到信念（belief）$b$。
4. 选音器按 $b$ 给每个候选音打分，取分最高的（argmax）。

HMM 的实时滤波（filter）：

$$b_{HMM}(z)\ \propto\ \Big[\sum_{z'}\hat b_{k-1}(z')\,A_{q(k)}(z'\!\to z)\Big]\cdot\exp\Big(\lambda\sum_{p=0}^{11}d_k(p)\,\log\tilde B(p\mid z)\Big),\qquad \lambda=1.5$$

混合：

$$b(z)=\frac{b_{HMM}(z)^{1-\alpha}\,b_{GRU}(z)^{\alpha}}{\sum_{z'=1}^{7}b_{HMM}(z')^{1-\alpha}\,b_{GRU}(z')^{\alpha}},\qquad \alpha=0.7,\qquad b_{GRU}=\tfrac{1}{3}\big(p_{\mathrm{full}}+p_{\mathrm{big}}+p_{\mathrm{s1}}\big)$$

选音（$m$ 是她听到的音，$h_{\mathrm{prev}}$ 是 Duet 上一个音）：

$$h^\ast=\arg\max_{h}\Big[\sum_{z}b(z)\,\mathbb{1}[h\in z]-0.3\,\mathbb{1}[\mathrm{oct}]-0.01\,|h-h_{\mathrm{prev}}|-0.4\,\mathbb{1}\big[|h-h_{\mathrm{prev}}|>7\big]-0.004\,|h-65|-0.6\,\mathbb{1}[\mathrm{par}]\Big]$$

$z$ 是 7 个和弦之一；$\hat b_{k-1}$ 是上一拍结束时的信念；$A_q$ 按小节位置 $q$ 分开；$d_k(p)$ 是这一拍里音级 $p$ 响了多久（她当前的音先算到拍末）；$\tilde B$ 是已经算进 10 % 听错的发射表（emission table）。候选 $h$ 只在 $53\le h\le\min(77,\,m-1)$ 且 $|m-h|\bmod 12\in\{0,3,4,7,8,9\}$ 里选。第一项是"$h$ 是当前和弦的和弦音的概率"；后 5 项是 m1 的权重（weight）w_oct、w_step、w_big、w_mid、w_par：别唱她的八度（oct）、少挪动、别大跳（big leap：超过 7 个半音, semitone）、别离音域中间 65 太远、别和她走平行五度/八度（parallel fifths / octaves，par；这一项要用到她上一个听到的音）；第 6 个权重 w_unis 只罚同音，因为 Duet 只唱在她听到的音下方，永远用不上。

<!-- src: futures/v3_nolatency/systems/m1_filter_v5lo.py (P: lam 1.5, cur_w 1.0, w_oct 0.3, w_step 0.01, w_big 0.4, w_mid 0.004, w_par 0.6, w_unis 1.0, below 1; Filter.belief, _beat_ll, _advance, step: _finish_beats(k) then append the note then belief(k); prev_m = her previous heard note for w_par); futures/v3_nolatency/systems/r2_mix_v2.py (Mix.step lines 27-29: ens.chord_probs first, then m1's step; Mix.belief); futures/v3_nolatency/systems/m3_seq_v4.py (Ens.chord_probs = mean of the 3 GRUs); duet_beat/theory.py (ALTO_MID = (53 + 77) / 2 = 65) -->

## 4. 数据和训练目标

**数据：** 每首歌有她的旋律（每个音的起止和 MIDI 音高）、主音、拍号、拍子网格，以及每一拍一个和弦标签：大调的 7 个三和弦（triad）I、ii、iii、IV、V、vi、vii°（罗马数字级数, Roman numerals）之一，或者无标签。模型的参数（parameters）只从训练集（train split）学。

| 数据集 | 训练集歌数 | val 歌数 |
|---|---:|---:|
| bach | 135 | 17 |
| folk | 511 | 170 |
| pop（Hooktheory） | 8149 | 300（只取前 300 首） |
| pop909 | 322 | 40 |
| 合计 | 9117 | 527 |

<!-- src: futures/v3_nolatency/results/review.json (m5_policy_v9 evidence: "train-set sizes (bach 135, folk 511, pop 8149, pop909 322 = bench3.train_sets)"; re-counted on the mini with bench3.train_sets() on 2026-11-01, heartbeat log runs_mini/v4_1101/hb/card_new_trainN.log); runs_mini/v4_1101/report/board.txt (val 17/170/300/40); duet_beat/bench.py (MAX_EVAL = 300, first 300 songs); futures/v3_nolatency/bench3.py train_sets (essen dropped); duet_beat/theory.py CHORDS; futures/v4_selfiter/PROGRAM.md §1.4 (pop = Hooktheory) -->

**① HMM（m1 原样，只是数数）：**

$$A_q(z'\!\to z)=\frac{n_q(z',z)+0.5}{\sum_{z''}\big(n_q(z',z'')+0.5\big)},\qquad B(p\mid z)=\frac{n(z,p)+1}{\sum_{p'}\big(n(z,p')+1\big)},\qquad \tilde B(p\mid z)=0.9\,B(p\mid z)+\frac{0.1}{12}$$

$n_q(z',z)$：训练集里相邻两拍从和弦 $z'$ 走到 $z$ 的次数，按小节位置分 6 张表（4/4 拍的第 1–4 拍各一张；其他拍号分"小节第 1 拍"和"其他拍"）；$n(z,p)$：和弦 $z$ 下音级 $p$ 一共唱了多少拍。每个数据集的总权重一样，pop 的 8149 首不会淹没 bach 的 135 首。这就是加了平滑（smoothing）的频率计数，不需要反复迭代训练。

<!-- src: futures/v3_nolatency/systems/m1_filter_v5lo.py Model (kappa 0.5; (B + 1) normalized; logB = log(0.9 Bc + 0.1/12); song weight (nmax / len(songs)) ** bal with bal = 1; tkey: 4/4 -> 0..3, other meters -> 4 (first beat of the bar) / 5) -->

**② 3 个 GRU（以前训练好的，这次没有重训）：**
- **输入**：每个音 24 个数——她听到的音级（相对主音，12 格的独热编码, one-hot）、音高高低 $(\mathrm{MIDI}-66)/12$、上一个音的长度（取 $\log_2$）、小节里第几拍（4 格）、在这一拍里的位置（$\sin$、$\cos$ 两个数）、是否正好在拍上、拍号（2 拍 / 3 拍 / 其他，3 格）。
- **结构**：$s_i=\mathrm{GRU}(s_{i-1},x_i)$（$x_i$ 先过一层输入层（input layer）），$p_\theta(\cdot\mid x_{1:i})=\mathrm{softmax}\big(g([s_i;\,x_i])\big)$；$g$ 是两层全连接层（fully connected layer），softmax（归一化指数函数）把 7 个分数变成概率。full：96 个单元（unit）、种子 0、训练 150 s；big：128 个单元、种子 0、270 s；s1：96 个单元、种子 1、200 s。
- **训练**：只用训练集的歌，输入按同样规则模拟 10 % 听错；每步 64 首歌（先等概率挑数据集再挑歌，长歌随机截 200 个音）；Adam 优化器（Adam optimizer），学习率（learning rate）0.002，梯度裁剪（gradient clipping）1.0，训练到规定秒数为止。
- **目标**：软标签（soft target）= 这个音会覆盖的那几拍的真和弦，按覆盖时长加权；损失（loss）是交叉熵（cross-entropy）：

$$y_i(z)=\frac{\sum_{k:\,c_k\ge 0}\mathrm{ov}(i,k)\,\mathbb{1}[c_k=z]}{\sum_{k:\,c_k\ge 0}\mathrm{ov}(i,k)},\qquad \mathcal{L}(\theta)=-\frac{1}{\sum_i w_i}\sum_i w_i\sum_{z=1}^{7}y_i(z)\,\log p_\theta\big(z\mid\tilde x_1,\dots,\tilde x_i\big)$$

两个求和都只算这个音盖住的、有标签的拍（$c_k\ge 0$；无标签的拍记作 −1），所以 $y_i$ 加起来正好是 1；$\tilde x$ 是加了模拟听错的特征；$w_i=1$ 表示这个音至少盖住一个有标签的拍（否则 $w_i=0$，不参加训练）。直观地说：让 GRU 给"她这个音响着时真正在响的和弦"尽量高的概率，而这正是 Duet 这个音要唱的那段时间。

<!-- src: futures/v3_nolatency/systems/m3_seq_v1.py (NF = 12+1+1+4+2+1+3 = 24; feat; Net: Linear+Tanh, GRU, Linear-ReLU-Linear on [y; x]; song_xy lines 67-74: overlap added only for covered beats with z[kk] >= 0, divided by that sum, W = 1 if any; train_net: 64 songs per step, dataset chosen uniformly then song, crop 200, Adam lr 2e-3, clip_grad_norm 1.0, wall-clock secs; loss line 107); futures/v3_nolatency/results/m3_seq.json (full: 96 units, 150 s); futures/v3_nolatency/systems/m3_seq_v2.py (big: 128 units, 270 s, default seed 0); futures/v3_nolatency/systems/m3_seq_s1.py (s1: seed 1, 200 s); futures/v3_nolatency/systems/m3_seq_v4.py _load -->

**③ 不是学出来的：** α 和选音器的 6 个权重。

**在 val 上调了什么、试了几个：**
- 这一轮新调的只有 α：试了 0.5、0.7、0.85，外加"α = 0.85 且 w_big = 0.8"，共 4 个变体（variant），选了 0.7（理由见第 5 节）。
- 整个自迭代循环（self-iteration loop，4 轮）在 val 上一共打分 28 个候选（candidate），另有约 40 + 27 + 15 个只做诊断的配置（也在 val 上算过分，但不是候选，没走 PROGRAM §2.3 的正式打分）。
- 继承下来的部分以前也是在 val 上选的：m1 的 HMM 设置和选音权重（例如 w_oct 在 0.6 / 0.3 / 0.2 / 0.15 / 0 里选了抄她少的 0.3）、3 个 GRU 的组合；GRU 的参数本身只见过训练集。

<!-- src: runs_mini/v4_1101/state_R2.json ideas E2 (r2_mix_v1..v4: a 0.5 / 0.7 / 0.85 / 0.85 + w_big 0.8); runs_mini/v4_1101/state_R4.json variants_scored_in_loop 28 and variants_note (about 40 + 27 + 15 diagnostics, not candidates); runs_mini/v4_1101/diag/, r3/, r4/ (the diagnostics' val music_nd, e.g. runs_mini/v4_1101/diag/why/why_a_v1.json); futures/v4_selfiter/PROGRAM.md §2.3 (how to score music_nd); futures/v3_nolatency/results/m1_filter.json other_systems m1_filter_v5lo note (w_oct curve 0.6 / 0.3 / 0.2 / 0.15 / 0); futures/v3_nolatency/m1_tune.py and futures/v3_nolatency/tune/m3_tune.py (score on eval_sets('val')); futures/v3_nolatency/results/review.json m3_seq_v5 (GRUs trained on train only) -->

## 5. 和谁比、差多少、为什么赢

| 系统 | 是什么 | 总分 | bach | folk | pop | pop909 |
|---|---|---:|---:|---:|---:|---:|
| third_below | 基线（baseline）：唱她下方按大调音阶数的三度（超出 53–77 时按八度移进来；pop/pop909 约 1.3 % 的音因此在她上方） | 0.5054 | 0.5429 | 0.4903 | 0.5121 | 0.4763 |
| m3_seq_v5 | 同样 3 个 GRU + 它自己的选音器（会抄她） | 0.6104 | 0.6241 | 0.6805 | 0.5713 | 0.5655 |
| m1_filter_v5lo | 老模型：HMM + 不抄她的选音器 | 0.6206 | 0.6655 | 0.6766 | 0.5601 | 0.5804 |
| r1_gru_chooser_v1 | 第 1 轮：只用 3 个 GRU 的平均 + m1 选音器 | 0.6440 | 0.6915 | 0.7134 | 0.5903 | 0.5808 |
| **r2_mix_v2** | **0.3 HMM × 0.7 GRU + m1 选音器** | **0.6444** | **0.6853** | **0.7126** | **0.5908** | **0.5891** |
| r3_style_mix_v3 | 风格感知 HMM（style-aware HMM）× GRU 各 0.5 + m1 选音器；最高分，不可上台 | 0.6532 | 0.7028 | 0.7178 | 0.5963 | 0.5959 |
| 告诉起音拍的真和弦 + m1 选音器 | 天花板；诊断，不是因果（causal）的 | 0.8403 | 0.8925 | 0.8899 | 0.7779 | 0.8009 |
| m4_oracle | 神谕（oracle，开卷系统）：告诉真和弦 + 它自己的选音规则 | 0.8388 | 0.8871 | 0.8784 | 0.7889 | 0.8009 |

<!-- src: runs_mini/v4_1101/nd/<name>/results/nd_<name>.json for third_below, m3_seq_v5, m1_filter_v5lo, r1_gru_chooser_v1, r2_mix_v2, r3_style_mix_v3; runs_mini/v4_1101/diag/diag_rt_v1.json headline_mean_of_4.music_nd.told and datasets.*.music_nd.told; futures/v3_nolatency/results/nd_m4_oracle.json; runs_mini/v4_1101/report/board.txt (m4_oracle row); third_below: futures/v3_nolatency/systems/third_below.py lines 11-20 (two scale steps down, else 3 semitones; while h < ALTO[0]: h += 12), recount runs_mini/v4_1101/hb/pD_F_third.log and pD_Z_third.log (above her heard note: bach 0/1073, folk 2/17326, pop 184/14388, pop909 169/13523) -->

**差多少：** 比 m1 总分 +0.0238（bach +0.0198、folk +0.0359、pop +0.0307、pop909 +0.0087），比 third_below +0.1390，离天花板 0.8403 还差 0.1959。换一套听错的随机抽样（seed 2）重打：0.6470 对 m1 的 0.6208，仍高 +0.0262。它通过了全部保留规则（keep rule）：总分至少 +0.005、folk 和 pop 各至少 +0.003、seed 2 至少 +0.003、没有数据集掉超过 0.01、抄她比例（double）≤ 0.08（它最高 0.0553）、大跳比例（big）和平行五/八度比例不超过 m1 + 0.01；审查（review）也全过：因果检查（causality probe）72/72、作弊检查（cheat grep）通过、0 个音出音域、从不高于她被听到的音（按她的真音，46019 个音里有 17 个在她上方，都是听错的音）。

<!-- src: nd files above (differences of the full-precision values); runs_mini/v4_1101/seed2/r2_mix_v2.json 0.647007; runs_mini/v4_1101/seed2/m1_filter_v5lo.json 0.620792; runs_mini/v4_1101/state_R2.json champion (double, big, par, seed2_delta_vs_m1 0.0262, review, keep.py pass); runs_mini/v4_1101/review/r2_mix_v2.json (d_range: sung 46019, above_share 0.0 = above her HEARD note); runs_mini/v4_1101/tools/review.py docstring (d) "above her HEARD note"; read-only recount runs_mini/v4_1101/hb/pD_F_above.log and pD_Z_above.log (r2_mix_v2: 17 of 46019 sung notes above her TRUE note, 0 above her heard note); futures/v4_selfiter/PROGRAM.md §4.2 -->

**为什么赢：和弦猜得更准。** 下表是每个音起音时，信念最大的和弦恰好是起音那一拍真和弦的比例（top-1 猜对率），以及给真和弦的平均对数概率（log probability，越接近 0 越好）：

| 猜和弦的方法 | bach | folk | pop | pop909 | 平均 | 真和弦平均 $\log p$ |
|---|---:|---:|---:|---:|---:|---:|
| m1 的 HMM | 0.471 | 0.571 | 0.357 | 0.334 | 0.434 | −1.491 |
| 3 个 GRU 的平均（只用 GRU） | 0.530 | 0.654 | 0.366 | 0.386 | 0.484 | −1.361 |
| **r2_mix_v2（相乘）** | **0.539** | **0.653** | **0.370** | **0.403** | **0.491** | **−1.327** |

参考：永远猜 I 级（主和弦, tonic chord）是 0.299。于是"和弦猜错"丢掉的时间从 0.2011 降到 0.1714。

<!-- src: runs_mini/v4_1101/diag/diag_champs_v1.json table_4dp (acc per dataset and headline for m1_filter_v5lo, r1_gru_chooser_v1, r2_mix_v2; wrong_chord_guess 0.2011 -> 0.1714); runs_mini/v4_1101/r3/C/diag_20261101_153920.json mean4.mean_logp_true (-1.491, -1.3614, -1.3272); runs_mini/v4_1101/diag/recheck/recheck_rt_extra_result_20261101_145317.json mean.share_I 0.2991 -->

**为什么要两个一起：**
- **GRU 记得更长，提升全靠它。** 历史截断（history ablation）实验：只给当前这一拍（GRU 只给当前音）时，冠军和 m1 几乎一样（0.6114 对 0.6101）；给最近 4 拍 / 16 拍 / 整首时，领先 +0.0141 / +0.0216 / +0.0238。HMM 把过去压成 7 个数，大约 16 拍以前的音自然就忘了；GRU 的隐藏状态记得更久。
- **HMM 让 GRU 别乱跳。** 只用 GRU 的 r1_gru_chooser_v1 总分差不多（0.6440），但 folk 的大跳比例 0.0212 超过上限 0.0195（m1 的 0.0095 + 0.01），没过保留规则。我们的解读：GRU 可能一个音就整个改判，选音器跟着跳；HMM 有转移表拖着，变化平稳。数据：α = 0 / 0.5 / 0.7 / 0.85 / 1 时，folk 大跳 0.0095 / 0.0163 / 0.0185 / 0.0205 / 0.0212，总分 0.6206 / 0.6428 / 0.6444 / 0.6454 / 0.6440，0.5 以上几乎是平的；0.85 只高 0.001 却 folk 大跳超标，所以 0.7 是通过全部检查里最高的。另一种修法是把 w_big 从 0.4 提到 0.6（r2_gru_chooser_v1，0.6433），也能过，但分数略低，还要改一个权重。
- **相乘 = 两边都同意才算。** 一个和弦只有两边都给它不低的概率才会胜出，任何一边认为"几乎不可能"就被否决；所以相乘后的平均猜对率和平均 $\log p$ 都比任何一边单独好（上表）。
- **为什么不用 GRU 自带的选音器（m3_seq_v5）：** 同样的 GRU 猜测，但它的选音器常抄她（double：bach 0.184、folk 0.118、pop 0.100、pop909 0.087），music_nd 把这算失败，只有 0.6104；换成 m1 的选音器后 double 最高 0.0553。
- **一行带过：** r3_style_mix_v3 分数最高（0.6532，seed 2 0.6527），但它的 HMM 一侧是 3 份 HMM（通用、只用 pop、只用 pop909）按拍号和她每拍的节奏实时猜出的风格加权，严格按 PROGRAM 第 4.3 条不算"那个 HMM"，记为"最高分、不可上台"。
- **和 Vincent 的耳朵一致：** 盲听（blind listening）时 Vincent 认为所有片段都合格，问题全在实时猜和弦；r2_mix_v2 改的正是这一步。

<!-- src: runs_mini/v4_1101/report/history_v2.txt (H = 0 / 4 / 16 / whole: r2_mix_v2 0.6114 / 0.6344 / 0.6422 / 0.6444, m1 0.6101 / 0.6203 / 0.6206 / 0.6206; H=0 = current beat for the HMM, current note for the GRUs; "the filter forgets notes older than about 16 beats on its own"); runs_mini/v4_1101/state_R1.json ideas E1 (folk big 0.02119 > 0.01949); runs_mini/v4_1101/state_R2.json ideas E1 (r2_gru_chooser_v1 0.6433, w_big 0.6) and E2 (trend; big_folk 0.0163 / 0.0185 / 0.0205); futures/v3_nolatency/leaderboard_v3.jsonl last val rows (folk big: m1_filter_v5lo 0.0095, r1_gru_chooser_v1 0.0212); futures/v3_nolatency/systems/m1_filter_v5lo.py (w_big 0.4); runs_mini/v4_1101/nd/m3_seq_v5/results/nd_m3_seq_v5.json (double); runs_mini/v4_1101/state_R3.json champion (r3_style_mix_v3 what, presentable_note: "from the meter and each beat's rhythm"); runs_mini/v4_1101/seed2/r3_style_mix_v3.json 0.652734; futures/v4_selfiter/listen/result.json -->

## 6. 哪里失败、为什么

r2_mix_v2 没得分的时间按原因分（每段只归进第一个符合的原因，顺序：起音拍没标签 → 和弦换在她的音中间 → 她下方没有好音 → 听错 → 和弦猜错 → 猜对仍没中；每列加起来是 1）：

| | bach | folk | pop | pop909 | 平均 |
|---|---:|---:|---:|---:|---:|
| 得分（music_nd） | 0.6853 | 0.7126 | 0.5908 | 0.5891 | 0.6444 |
| 和弦猜错：信念最大的不是起音拍的真和弦 | 0.1664 | 0.1419 | 0.1748 | 0.2027 | 0.1714 |
| 她下方没有好音：音域里低于她真音、又协和又是和弦音的音不存在 | 0.0207 | 0.0503 | 0.1184 | 0.0855 | 0.0687 |
| 听错：这个音被听成了别的音 | 0.0692 | 0.0641 | 0.0558 | 0.0681 | 0.0643 |
| 和弦换在她的音中间（held into a new chord） | 0.0229 | 0.0030 | 0.0384 | 0.0377 | 0.0255 |
| 和弦猜对了仍没得分 | 0.0355 | 0.0282 | 0.0208 | 0.0165 | 0.0252 |
| 起音那一拍没有和弦标签 | 0.0000 | 0.0000 | 0.0010 | 0.0005 | 0.0004 |

<!-- src: runs_mini/v4_1101/diag/diag_champs_v1.json systems.r2_mix_v2.datasets.*.miss_share and systems.r2_mix_v2.headline_mean_of_4.miss_share (buckets + music_nd = 1, buckets_plus_nd_minus_1 ~ 0); bucket order: runs_mini/v4_1101/tools/diag_rt.py -->

- **和弦猜错仍是最大的一块（0.1714）。** 根本原因：她在一个和弦下唱的那几个音，通常同时属于 2–3 个三和弦，旋律本身就分不出来——真和弦是"唯一最能覆盖她旋律"的三和弦的比例只有 bach 1 %、folk 22 %、pop 11 %、pop909 14 %（平均有 3.1 / 2.3 / 2.5 / 2.0 个三和弦并列）。剩下只能靠"哪个和弦更常见、和弦通常怎么走"去猜。
- **锁定（lock-on）很弱。** 同一个和弦下第 1 / 2 / 3 / 4 个及以后的音，猜对率是 0.468 / 0.477 / 0.511 / 0.521（m1 是 0.434 / 0.412 / 0.447 / 0.437，几乎是平的）；如果告诉它上一拍的真和弦，这条线能到 0.39 / 0.65 / 0.84 / 0.88。和弦刚换后的那个音最难猜（0.467，和弦没换时 0.501）。
- **听错的音：** 猜对率只有 0.349（听对的音是 0.507）；而且 Duet 是按听到的音挑协和音，和她的真音可能不协和。
- **她下方没有好音（0.0687）是设计决定：** Duet 只唱在她（听到的音）下方；其中 0.0422 的时间如果唱到她上方能找到好音。
- **和弦换在她的音中间（0.0255）：** Duet 的音和她的音一起保持、中途不改，pop 和 pop909 最多（0.0384、0.0377）。
- **猜对了仍没中（0.0252，m1 是 0.0193）：** 其中一部分是唱了她的八度——抄她的比例从 m1 的最高 0.0481 升到 0.0553（仍远低于上限 0.08）。

<!-- src: runs_mini/v4_1101/diag/why/why_d_v1.json datasets.*.all_spans (unique_best 0.0126 / 0.2246 / 0.1099 / 0.1380; mean_tied 3.12 / 2.33 / 2.49 / 2.02; computed on her TRUE melody); runs_mini/v4_1101/diag/diag_champs_v1.json table_4dp (r2_mix_v2 headline n1 0.468, n2 0.4767, n3 0.5113, n4+ 0.521, chg 0.4668, same 0.5008; m1_filter_v5lo n1..n4+ 0.4337 / 0.4123 / 0.4469 / 0.437; right_chord_still_missed 0.0193 -> 0.0252; double 0.0481 -> 0.0553), systems.r2_mix_v2.headline_mean_of_4.chord_acc (misheard 0.3495, heard_right 0.5068) and no_good_below_but_above_ok_share 0.0422; runs_mini/v4_1101/diag/diag_rt_v1.json chord_acc.past note_1..4_in_chord (0.3917 / 0.6537 / 0.8353 / 0.8791) -->

**最难的数据集：流行歌。** pop909 得分最低（0.5891），pop 几乎一样（0.5908）；pop 的和弦最难猜（猜对率 0.370），她的旋律落在和弦音上的时间最少（0.656，bach 是 0.916），她下方没有好音的时间最多（0.1184），连告诉真和弦的天花板也最低（0.7779）。

<!-- src: runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json; runs_mini/v4_1101/diag/diag_champs_v1.json (r2_mix_v2 pop acc 0.3697, pop no_good_note_below 0.1184); runs_mini/v4_1101/diag/why/why_b_v1.json evidence.*.chord_tone_share.true_all (bach 0.9164, pop 0.6560); runs_mini/v4_1101/diag/diag_rt_v1.json datasets.pop.music_nd.told 0.7779 -->

**标签本身的局限：** 七和弦（seventh chord）多出来的那个音也会被算成"不是和弦音"。我们的 7 个和弦全部由大调音阶（major scale）搭成，所以在小调（minor key）歌里，那个升高的、向上倾向主音的第 7 音（导音，leading tone）会被算成错——这是标签的局限，不是我们调参（parameter tuning）调出来的。

<!-- src: futures/v4_selfiter/PROGRAM.md §4.4 (known limit) and §5.4 (wording); runs_mini/v4_1101/report/board.txt (known label limit); duet_beat/theory.py (the seven diatonic triads of the major key) -->

**单首歌不一定更好：** 3 段 40 拍的试听片段（1 首儿歌 + 2 首巴赫众赞歌）里，r2_mix_v2 的片段得分是 0.583 / 0.591 / 0.653，m1 是 0.694 / 0.652 / 0.667。17 首 bach val 歌按整首比，r2_mix_v2 赢 9、平 1、输 7；只看开头 40 拍则赢 6、平 1、输 10。+0.0238 是 4 个数据集（共 527 首歌）各自 music_nd 平均之后的差，个别歌会反过来，放音频前心里要有数。

<!-- src: runs_mini/v4_1101/report/audio/index.md (clip music_nd); runs_mini/v4_1101/phaseD/A/clip_context.json summary (CHAMP = r2_mix_v2 in clip_context.py: song_level 9 / 1 / 7, clip_level 6 / 1 / 10); futures/v3_nolatency/rescore_nd.py line 28 (headline_nd = sum of the 4 dataset music_nd / 4); futures/v3_nolatency/bench3.py docstring ("Headline = mean of music over bach, folk, pop, pop909") -->

## 7. 下一步

1. **她的持续音（held note）跨过拍子边界（beat boundary）时，和弦很少换。** 第 4 轮发现：在她的音拖过去的拍子边界上，folk 的真和弦只换 3.6 %（转移表以为 22 %）；bach 在这种边界上换 26 %（表以为 22 %，差不多），但在全部边界上换 74 %（表只以为 38 %）——表不分"她的音拖着"和"新起音"的边界。让 HMM 的转移概率取决于"她的音是不是正拖过这个边界"，可能让 folk 和 bach 的猜和弦更准；还没在 HMM 滤波里试过。
2. **风格感知 HMM。** r3_style_mix_v3 已经把总分推到 0.6532（+0.0088）：HMM 一侧换成 3 份 HMM（通用、只用 pop、只用 pop909），按她的拍号和每拍的节奏实时猜出的风格加权，再和同样的 3 个 GRU 各占一半相乘。按 PROGRAM 第 4.3 条它不可上台，只记作"最高分、不可上台"；要用它，得另写卡片、另做测验。
3. **少抄她（w_oct）。** 冠军有 2.2–5.5 % 的时间唱她的同音或八度（music_nd 算失败）；把 w_oct（现在 0.3）调高，可能把其中一部分变成真正的和声。这是选音器的改动，要守住 double ≤ 0.08 和大跳上限。

<!-- src: runs_mini/v4_1101/state_R4.json ideas_round_4 E4 finding_for_later; runs_mini/v4_1101/r4/A/calib_20261101_160528.json calibration_val (folk held actual_change 0.0359 vs implied_change[0] = own_pooled = m1's table 0.2221; bach held 0.2613 vs 0.2185; bach all 0.7369 vs 0.3811); runs_mini/v4_1101/r4/A/calib_20261101_160528.json configs held_beat / held_cover (held tables used only inside E4's forecast); runs_mini/v4_1101/state_R3.json champion (r3_style_mix_v3 what, presentable_note); futures/v4_selfiter/PROGRAM.md §4.3; runs_mini/v4_1101/nd/r3_style_mix_v3/results/nd_r3_style_mix_v3.json; runs_mini/v4_1101/state_R2.json observations_not_acted_on (doubling, w_oct); runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json (double 0.0222 to 0.0553); futures/v3_nolatency/systems/m1_filter_v5lo.py (w_oct 0.3) -->

## 8. 教授最可能问的 10 个问题

1. **每听到她一个音，Duet 拿到什么、给出什么？**
   答：Duet 只知道这个音的起音时刻和听到的音高（10 % 会听错）、上一个音的长度、主音、拍号和拍子网格，没有和弦标签，也没有任何未来的信息。它在同一刻给出一个低于她（听到的音）的女中音音（MIDI 53–77），一直唱到她下一个音，每个音的计算约 0.15 毫秒。
   <!-- src: futures/v3_nolatency/bench3.py; duet_beat/bench.py heard(); duet_beat/theory.py ALTO; futures/v3_nolatency/leaderboard_v3.jsonl r2_mix_v2 us_per_note -->
2. **和老模型 m1 比，到底改了什么？**
   答：只改了"实时猜和弦"这一步：原来只用 HMM 的信念，现在用 HMM 和 3 个 GRU 平均的对数线性混合 $b\propto b_{HMM}^{0.3}b_{GRU}^{0.7}$。选音器和它的 6 个权重一个没动，把 α 设成 0 时它唱出的音和 m1 逐音相同（在 4 个数据集各 25 首训练歌上核对过）。
   <!-- src: futures/v3_nolatency/systems/r2_mix_v2.py; runs_mini/v4_1101/r2/B/check_mix_v1.json (a0_same_notes_as_m1 = notes on the first 25 train songs per dataset; r2_mix_v1 is the same code as r2_mix_v2 except ALPHA) -->
3. **music_nd 怎么算？为什么不算唱她的同音或八度？**
   答：跳过开头 4 拍，按她真唱的时间加权，Duet 的音要同时和她的真音协和（三度、五度、六度）、是这一拍真和弦的和弦音、又不是她的同音或八度才得分，4 个数据集各算一个再取平均就是总分。唱她的同音或八度只是重复她、不是和声，而原始分（raw music）会奖励这种抄写（m3_seq_v5 的原始分 0.720，music_nd 只有 0.6104），所以改用 music_nd。
   <!-- src: futures/v3_nolatency/rescore_nd.py; futures/v3_nolatency/bench3.py (J = 4); futures/v3_nolatency/leaderboard_v3.jsonl m3_seq_v5 last val headline 0.7199 (raw music, copying counted); runs_mini/v4_1101/nd/m3_seq_v5/results/nd_m3_seq_v5.json 0.6104 -->
4. **GRU 读什么、学什么？**
   答：每个音它读入 24 个数（听到的音级、音高高低、上一个音多长、小节里第几拍、在这一拍里的位置、是否在拍上、拍号），用隐藏状态记住到目前为止的整段旋律，输出 7 个和弦的概率。它在 9117 首训练歌上（输入里模拟 10 % 听错）用交叉熵学着预测"这个音会覆盖的那几拍的和弦"（按时长加权的软标签），3 个网络（96、128、96 个单元）的概率再取平均。
   <!-- src: futures/v3_nolatency/systems/m3_seq_v1.py; futures/v3_nolatency/results/review.json (train sizes 135 + 511 + 8149 + 322); futures/v3_nolatency/systems/r2_mix_v2.py build (_load full 96, big 128, s1 96) -->
5. **为什么要把 HMM 和 GRU 相乘，不只用 GRU？**
   答：相乘时一个和弦必须两边都觉得可能才会胜出，所以平均猜对率比任何一边单独都高（0.491，HMM 0.434，GRU 0.484）。只用 GRU 的版本总分差不多（0.6440），但它会突然改判和弦、带着 Duet 大跳，folk 的大跳比例 0.0212 超过上限 0.0195，混进 0.3 的 HMM 后降到 0.0185，通过了检查。
   <!-- src: runs_mini/v4_1101/diag/diag_champs_v1.json table_4dp headline acc; runs_mini/v4_1101/state_R1.json ideas E1; runs_mini/v4_1101/state_R2.json champion.big -->
6. **α = 0.7 怎么定的？会不会对 val 过拟合（overfitting）？**
   答：在 val 上只试了 0.5、0.7、0.85（外加一个同时改 w_big 的变体），总分 0.6428、0.6444、0.6454 几乎是平的，0.85 因 folk 大跳超标被排除，所以选了通过全部检查里最高的 0.7。曲线平说明结果不靠 α 的精确值，换一套听错抽样（seed 2）它仍比 m1 高 0.0262，但这些都是 val 上的数字，没有测试集做最后确认。
   <!-- src: runs_mini/v4_1101/state_R2.json ideas E2; runs_mini/v4_1101/seed2/r2_mix_v2.json; runs_mini/v4_1101/seed2/m1_filter_v5lo.json -->
7. **为什么不直接用 GRU 自带的选音器（m3_seq_v5）？**
   答：同样的 3 个 GRU，它自己的选音器经常抄她（double：bach 0.184、folk 0.118），music_nd 把这算失败，所以只有 0.6104，比 m1 还低。我们只借用 GRU 的和弦猜测，选音交给不抄她的 m1 选音器，double 降到最高 0.0553，总分升到 0.6444。
   <!-- src: runs_mini/v4_1101/nd/m3_seq_v5/results/nd_m3_seq_v5.json; runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json -->
8. **怎么保证它没偷看未来、没作弊？**
   答：因果检查在 24 首 val 歌、每首 3 个截断点上，把截断点以后的音高、时长和全部和弦标签随机打乱，截断点之前的输出 72/72 完全不变。代码检查确认它建模只用训练集、每一步只用 step() 收到的输入（GRU 每个音只前进一次）；0 个音超出女中音音域，也从不高于她被听到的音（按她的真音，46019 个音里有 17 个在她上方，都是听错的音）。
   <!-- src: futures/v4_selfiter/PROGRAM.md §4.1 reviewer (a), (b), (d); runs_mini/v4_1101/review/r2_mix_v2.json (a_causal 72 checks, 0 mismatches; b_grep ok; d_range out_of_alto 0, above_share 0.0 = above her HEARD note); runs_mini/v4_1101/tools/review.py docstring (d); read-only recount runs_mini/v4_1101/hb/pD_F_above.log and pD_Z_above.log (17 of 46019 above her TRUE note); runs_mini/v4_1101/diag/diag_champs_v1.json (one_belief_per_note_all true) -->
9. **离"告诉你和弦"的天花板还差多少？差在哪？**
   答：同一个选音器如果被告诉起音那一拍的真和弦能到 0.8403（另一个神谕系统 m4_oracle 是 0.8388），我们是 0.6444，还差 0.196。最大的一块是和弦猜错（0.1714 的时间），其次是她下方没有好音（0.0687）和听错（0.0643）。
   <!-- src: runs_mini/v4_1101/diag/diag_rt_v1.json headline_mean_of_4.music_nd.told; futures/v3_nolatency/results/nd_m4_oracle.json; runs_mini/v4_1101/diag/diag_champs_v1.json systems.r2_mix_v2.headline_mean_of_4.miss_share -->
10. **哪个数据集最难？为什么？**
    答：流行歌最难：pop909 得分最低（0.5891），pop 几乎一样（0.5908），而 pop 的和弦猜对率只有 0.370。pop 的旋律只有约 66 % 的时间落在和弦音上（pop909 约 72 %，bach 约 92 %），她下方找不到好音的时间有 0.1184，连告诉真和弦的天花板也只有 0.7779。
    <!-- src: runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json; runs_mini/v4_1101/diag/diag_champs_v1.json (pop acc 0.3697, no_good_note_below 0.1184); runs_mini/v4_1101/diag/why/why_b_v1.json chord_tone_share.true_all (bach 0.9164, pop 0.6560, pop909 0.7212); runs_mini/v4_1101/diag/diag_rt_v1.json datasets.pop.music_nd.told -->

## 9. 术语表

每条一个术语：中文（English）：一句解释；生活例子。

- **验证集（validation split, val）**：调设置、比较版本用的歌；生活例子：模拟考。
- **训练集（train split）**：模型学参数用的歌；生活例子：课本上的练习题。
- **测试集（test split）**：最后才看、从不拿来调的歌，这个包没有；生活例子：正式考试。
- **数据集（dataset）**：同一来源的一批歌，这里是 bach（巴赫众赞歌）、folk（民歌）、pop 和 pop909（两套流行歌），每首都有旋律和每拍的和弦标签；生活例子：四本风格不同的歌本。
- **拍（beat）**：音乐均匀的时间格子；生活例子：听歌时点头的那一下。
- **小节（bar）**：几拍组成的一组，第一拍最重；生活例子：华尔兹"嘭嚓嚓"三拍一组。
- **拍号（meter）**：一小节有几拍；生活例子：进行曲数"一二、一二"，华尔兹数"一二三"。
- **拍子网格（beat grid）**：标出每一拍什么时候来、是小节里的第几拍；生活例子：节拍器的滴答声和重音。
- **起音（onset）**：一个音开始的时刻；生活例子：按下琴键的那一下。
- **MIDI 音高（MIDI note number）**：用整数表示音高，每加 1 高一个半音；生活例子：给钢琴键从左到右编号。
- **半音（semitone）**：钢琴上相邻两个键（含黑键）的距离；生活例子：紧挨着的白键和黑键。
- **音级（pitch class）**：不管在哪个八度的音名，一共 12 个；生活例子：男声和女声唱的 do 是同一个音级。
- **听错（mishearing）**：每个音有 10 % 的概率被当成另外 11 个音级之一；生活例子：在嘈杂的饭馆把"十四"听成"四十"。
- **主音（tonic）**：一个调的"家"；生活例子：一首歌最后落回来、让人觉得"唱完了"的那个音。
- **大调（major key）**：以大调音阶为基础、听起来明亮的调；生活例子：多数儿歌的明亮感。
- **大调音阶（major scale）**：do re mi fa sol la si 七个音；生活例子：从 C 开始只弹白键弹一圈。
- **小调（minor key）**：听起来更暗、偏伤感的调；生活例子：电影里悲伤场景的配乐。
- **和弦（chord）**：同时响的一组音；生活例子：吉他按一个把位扫出来的声音。
- **三和弦（triad）**：隔一个音叠起来的 3 个音，如 C–E–G；生活例子：钢琴上 do、mi、sol 一起按。
- **七和弦（seventh chord）**：三和弦上再叠一个音的四音和弦；生活例子：爵士乐里"带点味道"的和弦。
- **罗马数字级数（Roman numerals）**：I、ii、iii、IV、V、vi、vii° 表示建在大调第 1–7 个音上的三和弦；生活例子：弹唱的人说的"1-4-5 级和弦"。
- **主和弦（tonic chord）**：I 级和弦，以主音为底的三和弦；生活例子：歌曲结尾"回到家"的那个和弦。
- **和弦音（chord tone）**：属于当前和弦的音；生活例子：C 和弦响着时唱 C、E、G 都"对"。
- **和弦标签（chord label）**：数据里给每一拍标好的真和弦，Duet 运行时看不到；生活例子：练习册后面的答案页。
- **协和（consonant）**：两个音一起听很稳，这里指相差 0、3、4、7、8、9 个半音（同音/八度、三度、五度、六度）；生活例子：两人合唱时"合得上"的声音。
- **和声（harmony）**：和旋律同时响、给它配色的音；生活例子：唱生日歌时有人在下面唱另一条线。
- **同音（unison）**：完全一样高的两个音；生活例子：两人唱一模一样的音高。
- **八度（octave）**：同名、一高一低、差 12 个半音的两个音；生活例子：男生女生"齐唱"其实差一个八度。
- **抄她（doubling）**：Duet 唱她的同音或八度，只是重复她、不是和声；生活例子：合唱时有人不唱自己那条线、跟着唱主旋律。
- **抄她比例 double（doubling share）**：她唱歌的时间里 Duet 在抄她的比例；生活例子：一节课里只是跟着老师复读的时间占多少。
- **公平分 music_nd（fair score）**：Duet 的音"和她的真音协和、是真和弦的和弦音、不是抄她"三条都满足的时间比例；生活例子：三项都对才给分的考试。
- **原始分（raw music）**：旧评分，抄她也算对；生活例子：抄同桌答案也给分的考试。
- **总分（headline）**：4 个数据集各自 music_nd 的平均（每个数据集一样重）；生活例子：四门课的平均分，不管每门课多少学生。
- **女中音音域（alto range）**：MIDI 53–77（F3–F5），Duet 能唱的高低范围；生活例子：合唱团女中音能唱的最低到最高。
- **零延迟（zero latency）**：她起音的同一刻就听到并回答；生活例子：两人击掌，手同时到。
- **实时猜和弦（real-time chord guess）**：只凭到现在为止听到的音，猜此刻是哪个和弦；生活例子：收音机放到一半，猜这是哪首歌。
- **信念（belief）**：此刻对 7 个和弦各有几成把握，加起来是 1；生活例子：天气 App 说"晴的可能大、下雨的可能小"。
- **隐马尔可夫模型（HMM, hidden Markov model）**：看不见的状态（和弦）按转移表一步步变，每步产生看得见的东西（她的音）；生活例子：从邻居家哪间屋亮灯猜他们在干什么，还知道"吃完饭一般去洗碗"。
- **滤波（filter）**：每结束一拍，用"上一拍的信念 × 转移表 × 这一拍的证据"更新一次；生活例子：开车时用刚才的位置加上眼前的路牌，不断修正"我在哪"。
- **转移表（transition table）**：上一拍是某和弦时，这一拍是各和弦的概率；生活例子：今天晴，明天多半还晴。
- **发射表（emission table）**：某和弦下她唱各音级的概率；生活例子：川菜馆里多半能看到辣椒。
- **平滑（smoothing）**：数次数时每格先加一个小数，没见过的情况概率也不为 0；生活例子：新开的店没有评价，先默认给个中等分。
- **参数（parameters）**：模型里靠训练调出来的数；生活例子：菜谱里每样调料放几克。
- **调参（parameter tuning）**：反复改模型的几个设置，挑分数最高的；生活例子：拧收音机旋钮直到声音最清楚。
- **门控循环单元（GRU, gated recurrent unit）**：一个音一个音往下读的神经网络，用"门"决定记住多少、忘掉多少；生活例子：听故事时脑子里不断更新的剧情摘要。
- **单元（unit）**：GRU 记忆里的一个数，96 或 128 个单元就是记忆有 96 或 128 个数；生活例子：笔记本有多少行可以记东西。
- **隐藏状态（hidden state）**：GRU 里那串数，就是它的记忆；生活例子：小说读到一半时你记着的人物关系。
- **特征（feature）**：把一个音写成的一串数；生活例子：体检表上的身高、体重、血压。
- **独热编码（one-hot）**：12 格里只有一格是 1、其余是 0 的写法；生活例子：问卷里 12 个选项只能勾一个。
- **输入层（input layer）**：网络最前面的一层，先把 24 个特征换算成网络内部用的一串数；生活例子：翻译先把外语句子转成自己的话，再往下想。
- **全连接层（fully connected layer）**：每个输入都连到每个输出、各乘一个参数再相加的一层；生活例子：每个评委给每个选手打分再汇总。
- **归一化指数函数（softmax）**：把 7 个分数变成都为正、加起来是 1 的概率；生活例子：把候选人的票数换算成百分比。
- **对数线性混合（log-linear mix）**：两个概率分别取 $1-\alpha$、$\alpha$ 次方再相乘；生活例子：两位评委的分相乘来排名，任何一位给 0 分就出局。
- **归一化（normalize）**：除以总和，让 7 个数加起来是 1；生活例子：把各组的人数换算成占全班的百分比。
- **软标签（soft target）**：目标按时长分给几个和弦，比如一半 I、一半 V；生活例子：拼车按各自坐的路程分摊车费。
- **交叉熵（cross-entropy）**：模型给真实答案的概率越低罚得越重（取 $-\log$）；生活例子：天气预报说"几乎不会下雨"结果下了，大家特别不满。
- **损失（loss）**：训练时要降下来的"扣分"总数，这里就是交叉熵；生活例子：考试被扣掉的总分。
- **Adam 优化器（Adam optimizer）**：按误差一小步一小步自动调参数的方法；生活例子：蒙着眼顺着坡一步步往下走，步子随路况调整。
- **学习率（learning rate）**：每一步调多大；生活例子：调收音机旋钮时每次拧多少。
- **梯度裁剪（gradient clipping）**：限制一步最多调多大，免得一下调过头；生活例子：开车限速。
- **随机种子（random seed）**：决定"随机"结果的起点，同一个种子结果可重复；生活例子：约定好的洗牌方法。
- **seed 2（second seed）**：换一个随机种子、也就是换一套听错的抽样，再打一次分，看结果稳不稳；生活例子：同一副牌换一种洗法再发一次。
- **选音器（chooser）**：给每个候选音打分、挑分最高的规则；生活例子：点菜时按"好吃 − 太贵 − 和上一道太像"打分。
- **权重（weight）**：每项代价前面乘的系数，决定它有多重要；生活例子：总评里期末考占 60 %、平时占 40 %。
- **取最大（argmax）**：在所有候选里挑分数最高的那一个；生活例子：选票最多的人当选。
- **大跳（big leap）**：Duet 相邻两个音差超过 7 个半音；生活例子：说话声调突然从很低窜到很高。
- **大跳比例 big（big-leap share）**：Duet 换音里大跳占的比例；生活例子：一段路里急转弯占几成。
- **平行五度/八度（parallel fifths / octaves）**：两条旋律线同方向移动、一直保持五度或八度，传统写法要避免；生活例子：两人动作完全同步，看起来像一个人。
- **基准测试（benchmark）**：固定的题目和评分规则，这里是 bench3；生活例子：统一的考试卷。
- **top-1 猜对率（top-1 accuracy）**：概率最大的那个和弦正好是真和弦的比例；生活例子：第一个答案就猜中的比例。
- **对数概率（log probability）**：给真和弦的概率取对数，越接近 0 越好；生活例子：答对时你押了多少"信心"。
- **基线（baseline）**：最简单的参照系统，这里是唱她下方按大调音阶数的三度；生活例子：及格线。
- **天花板（ceiling）**：直接告诉系统真和弦时能拿到的分，现实里拿不到；生活例子：开卷考试的分数。
- **神谕（oracle，开卷系统）**：能看到真和弦的系统，只用来量天花板；生活例子：拿着答案对答案。
- **因果（causal）**：只用到目前为止的信息、不偷看未来；生活例子：现场直播，不能事后剪辑。
- **因果检查（causality probe）**：把截断点以后的"未来"打乱，看之前的输出变不变；生活例子：考试中途换掉后面的题，前面已交的答案不该变。
- **审查（review）**：候选成为冠军前对代码和输出的检查；生活例子：新药上市前的审批。
- **作弊检查（cheat grep）**：在代码里搜有没有读 val 答案或其他不该读的文件；生活例子：考前检查有没有夹带小抄。
- **保留规则（keep rule）**：新候选要成为冠军必须同时通过的全部检查；生活例子：转正考核，每一项都要达标。
- **可上台的冠军（presentable champion）**：只由"一个猜和弦器 + m1 选音器"组成、三句话能讲清的冠军；生活例子：只换了一个零件的车，好解释它为什么更快。
- **风格感知 HMM（style-aware HMM）**：几份按风格分开数出来的 HMM，按实时猜出的风格加权；生活例子：先听口音猜对方是哪里人，再按当地习惯理解他的话。
- **历史截断（history ablation）**：只让模型看最近 H 拍的旋律，看分数掉多少；生活例子：只让你读小说最后几页，看你能猜对多少剧情。
- **自迭代循环（self-iteration loop）**：电脑一轮轮自己提出改动、打分、按规则决定留不留；生活例子：厨师每晚试几道新菜，只留下明显更好吃的。
- **候选（candidate）**：循环里正式打分、再按保留规则判去留的一个改动版本；生活例子：参加选拔赛的选手。
- **变体（variant）**：同一个点子的不同设置（比如 α 取不同的值）；生活例子：同一道菜少放盐、多放盐的几个版本。
- **盲听（blind listening）**：不知道哪段是哪个系统时比较好不好听；生活例子：可乐盲测。
- **锁定（lock-on）**：同一个和弦下听得越久猜得越准；生活例子：收音机里一首歌越听越确定是哪首。
- **持续音（held note）**：她的一个音从一拍拖到下一拍；生活例子：唱歌时把一个字拖长好几拍。
- **拍子边界（beat boundary）**：相邻两拍之间的那条线；生活例子：尺子上的刻度线。
- **和弦换在她的音中间（held into a new chord）**：她的音还拖着，伴奏的和弦已经换了，Duet 跟着拖的音可能不再合适；生活例子：一个字拖长时伴奏已经变了。
- **过拟合（overfitting）**：设置调得太贴合某一批数据，换一批就不灵；生活例子：只背了模拟卷的答案，正式考换了题就不会。

<!-- src: duet_beat/theory.py (MAJOR, CT, CONSONANT = {0,3,4,7,8,9}, ALTO = (53, 77) "F3..F5"); duet_beat/bench.py heard() (10 %, other 11 pitch classes); futures/v3_nolatency/bench3.py (big = share of moves over 7 semitones; par); futures/v3_nolatency/systems/m1_filter_v5lo.py; futures/v3_nolatency/systems/m3_seq_v1.py (one-hot pc; input Linear+Tanh; 96 / 128 units via m3_seq_v4._load); futures/v3_nolatency/systems/third_below.py; futures/v3_nolatency/rescore_nd.py (headline = mean of the 4 datasets); futures/v4_selfiter/PROGRAM.md §4.2 (keep rule), §4.3 (presentable) -->
