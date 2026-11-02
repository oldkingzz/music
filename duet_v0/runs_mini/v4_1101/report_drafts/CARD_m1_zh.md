# m1_filter_v5lo 讲解卡（旧的周二模型）

<!-- 草稿：CARD_m1_zh.md，run v4_1101，Phase D，2026-11-01。所有 src 路径都相对 ABS = /Users/san/claude_sandbox/music/duet_v0。
缩写：nd/X = runs_mini/v4_1101/nd/X/results/nd_X.json；diag_rt = runs_mini/v4_1101/diag/diag_rt_v1.json；
diag_champs = runs_mini/v4_1101/diag/diag_champs_v1.json；extra = runs_mini/v4_1101/diag/recheck/recheck_rt_extra_result_20261101_145317.json；
why_X = runs_mini/v4_1101/diag/why/why_X_v1.json；lb:N = futures/v3_nolatency/leaderboard_v3.jsonl 第 N 行，即该系统最后一条 val 记录
（169 = m1_filter_v5lo，170 = third_below，171 = m3_seq_v5，都写于 2026-11-01 11:40）。 -->

读法：验证集（validation set，下文写 val）是用来调参（parameter tuning）和打分的歌，测试集（test set）是留到最后才看的歌。全部是 val 上的数字，也在 val 上调（这个包没有 test）。总分（headline）= 四个数据集（dataset）分数的平均：bach（巴赫众赞歌，Bach chorales）17 首，folk（民歌）170 首，pop 300 首，pop909 40 首（后两个是流行歌）。

<!-- src: futures/v4_selfiter/PROGRAM.md §1.3（eval_sets('test') raises KeyError）；futures/v3_nolatency/bench3.py docstring "Headline = mean of 'music' over bach, folk, pop, pop909"；runs_mini/v4_1101/report/board.txt 第 10 行 "bach 17, folk 170, pop 300, pop909 40 val songs"；runs_mini/v4_1101/diag/recheck/recheck_rt_result_20261101_145137.json → per_ds.*.songs = 17/170/300/40 -->

## 1. 一句话

m1_filter_v5lo 是 Duet 的旧版实时（real-time）和声（harmony）模型：用隐马尔可夫模型（Hidden Markov Model, HMM）的前向滤波（forward filter）边听边猜和弦（chord），再用一个只唱在她下面、尽量不跟她唱同一个音的选音器（chooser）配一个音，val 上音乐分（music_nd）是 0.6206。

<!-- src: futures/v3_nolatency/systems/m1_filter_v5lo.py（docstring；P['below'] = 1，P['w_oct'] = 0.3）；nd/m1_filter_v5lo → headline_nd 0.620637625406759 -->

## 2. 三句话讲清楚

她每开始唱一个音，Duet 就用 HMM 前向滤波更新"此刻是大调（major key）7 个三和弦（triad）里的哪一个"的概率，刚开始的这个音也算进去。然后 Duet 只在她听到的音下面、只在协和音程（consonant interval）里挑，选"是和弦内音（chord tone）的概率"减去抄她（doubling）和声部进行（voice leading）惩罚（penalty）后得分最高的音，一直唱到她的下一个音。它在 val 上得 0.6206，高于"在她下方唱三度（third）"规则的 0.5054 和会抄她的门控循环单元（gated recurrent unit, GRU）系统的 0.6104，但离"直接告诉它真和弦"的上限（ceiling）0.8403 还差 0.22，差距主要在实时猜和弦。

<!-- src: systems/m1_filter_v5lo.py → step()（先把当前音放进 self.notes，再调用 belief(k)；hi = min(ALTO[1], midi - 1)；iv in CONSONANT）；nd/m1_filter_v5lo 0.6206；nd/third_below 0.5054；nd/m3_seq_v5 0.6104；diag_rt → headline_mean_of_4.music_nd.told 0.8403194；0.22 = 0.8403 - 0.6206 -->

## 3. 输入和输出（input and output）

她每开始唱一个音（第 $i$ 个），Duet 拿到：
- 这个音的起始时间（onset），单位是拍（beat）；以及听到的 MIDI 音高（MIDI note number）。听错（mishearing）：每个音独立地有 10 % 的机会被换成另外 11 个音级（pitch class）之一，放在离原音最近的八度（octave）里；每首歌的随机种子（random seed）固定。
- 上一个音的长度（她开始唱这个音时，才知道上一个音唱了多久）。
- 调（key）的主音（tonic）、拍号（meter，每小节（bar）几拍）和拍子网格（beat grid，每一拍在小节里排第几）。
- Duet 自己的上一个音（只用来让自己的线条平顺，不是她的旋律（melody））。
- 不给：任何和弦标签（label）；任何未来的东西（后面的音、这个音自己会唱多久）。

Duet 回答：一个中音声部（alto）的 MIDI 音，范围 53–77，在她这个音开始的同一刻唱起，一直唱到她下一个音开始；她下面如果一个可用的协和音都没有，就不唱（val 上 46310 个音里有 291 个）。这就是零延迟（zero latency）：答案和她的音同时出来，每个音只算 11.9–13.5 微秒（Mac mini M4）。

没偷看未来：因果检查（causality probe）在 24 首 val 歌 × 3 个切点上，把切点之后的音高、时长和全部和弦标签随机打乱，切点之前 Duet 的输出 72 次检查全部不变。打分用她的真实音（不是听到的）和每一拍的真和弦，从第 4 拍开始算。

<!-- src: futures/v3_nolatency/bench3.py → docstring（"at her note i it knows the pitches and onsets of notes 0..i and the durations of notes 0..i-1, plus the key (tonic), the meter and the beat grid. No chord labels"）、run_song()、J = 4；duet_beat/bench.py → NOISE = 0.10、heard()（randint(1, 11)，最近八度，seed ^ 5160）；duet_beat/theory.py → ALTO = (53, 77)；runs_mini/v4_1101/review/m1_filter_v5lo.json → a_causal.checks 72、mismatches []、d_range.silent 291、d_range.sung 46019；diag_champs → systems.m1_filter_v5lo.notes_fed 46310；lb:169 → rows.*.us_per_note 13.47/11.90/11.96/12.82；runs_mini/v4_1101/state_R0.json → machine.cpu "Apple M4"；PROGRAM.md §4.1(a)（24 val songs × 3 cut points） -->

## 4. 数据和训练目标（data and training objective）

训练集（training set）是带和弦标签（每拍一个，少数拍没有）的干净歌（没有听错）：

| 数据集 | train 歌数 $N_d$ | 每首歌在计数里算几次 $N_{\max}/N_d$ | val 歌数 |
|---|---:|---:|---:|
| bach | 135 | 60.4 | 17 |
| folk | 511 | 15.9 | 170 |
| pop | 8149 | 1.0 | 300 |
| pop909 | 322 | 25.3 | 40 |

<!-- src: train 歌数 = len(bench3.train_sets()[ds])（bench3 去掉了没有和弦的 essen），由只读脚本统计 duet_beat/data/bach_beats.json → split.train 与 duet_beat/bench/data/{folk,pop,pop909}.json → train 得到（2026-11-01 16:55，heartbeat runs_mini/v4_1101/hb/card_m1_facts2.log，rc 0；facts1 的 rc 120 只是输出接了 head 造成的断管；没有现成结果文件记这几个数，核对请重算）；N_max / N_d：systems/m1_filter_v5lo.py → Model.__init__ 里 wd = (nmax / len(songs)) ** P['bal']，P['bal'] = 1，nmax = 8149；val 歌数见文首 src -->

学什么（全靠数次数，没有梯度下降（gradient descent））：
1. 转移表（transition table）$A_\tau(z'\to z)$：上一拍是和弦 $z'$ 时，这一拍是 $z$ 的概率。按这一拍在小节里的位置 $\tau$ 分 6 张：4/4 拍每拍一张，其他拍号分强拍（downbeat）和其他拍两张。学到的规律是"多半在第 1 拍换和弦"：4/4 拍里留在原和弦的概率（7 个和弦平均），第 1 拍 0.17，第 2 拍 0.77，第 3 拍 0.51，第 4 拍 0.82。
2. 发射概率（emission probability）$B(c\mid z)$：和弦 $z$ 下，她的旋律在音级 $c$ 上唱了多久（按拍计）。
3. 开头概率 $\pi(z)$：每首歌第一个有标签的拍是哪个和弦。

训练目标（training objective）是加权的最大似然（maximum likelihood）：

$$\max_{\pi,A,B}\ \sum_{d}\frac{N_{\max}}{N_d}\sum_{s\in d}\Big[\log\pi(z_{s,1})+\sum_{k}\log A_{\tau_k}(z_{k-1}\to z_k)+\sum_{k}\sum_{c=0}^{11}u_k(c)\,\log B(c\mid z_k)\Big]$$

$s$ 是数据集 $d$ 里的一首歌，$N_{\max}=8149$，$u_k(c)$ 是第 $k$ 拍里音级 $c$ 唱了几拍。使它最大的解就是"次数除以总数"；代码在除之前给每个格子加伪计数（pseudo-count）（转移 +0.5，发射和开头 +1），免得没见过的组合概率为 0：

$$A_\tau(i\to j)=\frac{n_\tau(i\to j)+0.5}{\sum_{j'}n_\tau(i\to j')+3.5},\qquad B(c\mid z)=\frac{n(z,c)+1}{\sum_{c'}n(z,c')+12}$$

$n$ 是乘过权重的次数。权重 $N_{\max}/N_d$ 就是数据集平衡（dataset balancing）：按歌数算，四个数据集在计数里一样重，8149 首 pop 不会淹没 135 首 bach。

<!-- src: systems/m1_filter_v5lo.py → S = 7；tkey()（M == 4 时 pos % 4，否则强拍 4、其他拍 5）；Model.__init__（A[tkey, z[k-1], z[k]] += wd；B[z[k]] += wd * obs[k]；pi 记第一个有标签的拍；A = (A + kappa) / sum，kappa = 0.5；Bc = (B + 1) / sum；pi = (pi + 1) / sum）；留在原和弦的概率 = extra → A_self_prob {"0": 0.1706, "1": 0.7704, "2": 0.5098, "3": 0.8230}（= mean(diag(A[tkey]))；why_a → pstay_tables.m1_pooled_bal1 相同） -->

实际演唱时分两步：

**第 1 步：猜和弦（前向滤波）。** 每一拍先用转移表把上一拍的信念（belief）往前推，再乘上这一拍她唱的音给的线索：

$$b_k(z)\ \propto\ \Big[\sum_{z'}b_{k-1}(z')\,A_{\tau_k}(z'\to z)\Big]\cdot\exp\Big(\lambda\sum_{c}v_k(c)\,\log\big(0.9\,B(c\mid z)+\tfrac{0.1}{12}\big)\Big),\qquad\lambda=1.5$$

- $v_k(c)$：到此刻为止第 $k$ 拍里她（听到的）音级 $c$ 唱了多久；她正在唱的音先当作会唱到这一拍结束，选音之前就算进去。如果当前这一拍里已经听到的音（包括这个音）都不算，music_nd 是 0.607、和弦准确率（accuracy）0.338（现在是 0.6206 和 0.434）。
- $0.9B+0.1/12$：给每个音级留一点余量，容忍 10 % 听错。$\lambda$ 是证据权重（evidence weight）：让她唱的音比"平时的习惯"更有分量。
- 信念里概率最大的那个和弦叫首选和弦（top guess）。

**第 2 步：选音（选音器）。** $m$ = 她听到的音，$h_{\text{prev}}$ = Duet 上一个音，$\mathrm{CT}(z)$ = 和弦 $z$ 的三个音，$\mathbb 1[\cdot]$ 条件成立记 1：

$$h^*=\arg\max_{h\in\mathcal H}\Big[\sum_{z}b(z)\,\mathbb 1[h\in\mathrm{CT}(z)]-0.3\,\mathbb 1[\text{八度}]-0.01\,|h-h_{\text{prev}}|-0.4\,\mathbb 1\big[|h-h_{\text{prev}}|>7\big]-0.004\,|h-65|-0.6\,\mathbb 1[\text{平行五度/八度}]\Big]$$

- $\mathcal H$：53 到 $\min(77,\,m-1)$ 之间、和 $m$ 协和（相差 3、4、7、8、9 个半音（semitone），再隔几个八度也算，或者正好隔几个八度）的音；$\mathcal H$ 是空的就不唱。
- 6 个惩罚的权重：八度 w_oct 0.3；每移动一个半音 w_step 0.01；超过 7 个半音的大跳（leap）w_big 0.4；离 alto 中间（65）每个半音 w_mid 0.004；平行五度/八度（parallel fifths/octaves）w_par 0.6；同度（unison）w_unis 1.0——最后这个永远用不上，因为 Duet 只唱在她下面。

<!-- src: systems/m1_filter_v5lo.py → _beat_ll()（P['lam'] * (logB @ v)；当前音 off 为 None 时算到 k + 1）、_advance()、belief()（cur_w 1.0）、step()（pfit = b @ CTM；hi = min(ALTO[1], midi - 1)；iv == 0 时减 w_oct；w_step * d + w_big * (d > 7) + w_mid * |h - ALTO_MID|；w_par）；Model → logB = log((1 - 0.1) * Bc + 0.1 / 12)；P：lam 1.5、w_step 0.01、w_big 0.4、w_par 0.6、w_mid 0.004、w_oct 0.3、w_unis 1.0、below 1；duet_beat/theory.py → CONSONANT {0, 3, 4, 7, 8, 9}、ALTO (53, 77)、ALTO_MID 65；当前拍证据乘 0 = why_a → variants.C_curw0.rt.head（music_nd 0.6070，acc 0.3381；runs_mini/v4_1101/diag/why/why_lib.py：ll = self._beat_ll(k) * self.cur_w）；diag_rt → headline_mean_of_4.chord_acc.rt.all.acc 0.4335 -->

在 val 上调了什么、试了多少：
- 调过（笔记本电脑上，2026-10-31）：发射里加入 10 % 听错、四个数据集平衡、w_step 0.04→0.01（旧分数（raw music，抄她也算对）上分别 +0.008、+0.009、+0.015）；λ 从第一版的 1.0 改成 1.5；w_oct 试了 0.6、0.3、0.2、0.15、0 五个值；当前音的权重试过 0.7 和 1.3，都更差。没动过：转移伪计数 0.5、w_big、w_par、w_mid、w_unis。
- 名字里的 lo：v5 的 w_oct 是 0.15，旧分数更高（0.687），但抄她率（double）约 0.2；v5lo 改成 0.3，旧分数 0.654，抄她率降到 0.027–0.048。
- 数量：笔记本电脑上有 8 个 m1 版本文件（v1–v7 和 v5lo），3 个上过记录表，另记了 7 个试过没用的想法（如提前看下一拍、记住几小节前的和弦）；临时开关扫参数的确切次数没有记录。Mac mini 上自动迭代（self-iteration）的第 1 轮又在 val 上分析了约 40 个 m1 相关配置（不是候选）：不偷看答案、只改一个 HMM 的设置，最好是 λ = 1.0 的 0.6233，只高 0.003，达不到保留规则（keep rule）要求的 +0.005。

<!-- src: futures/v3_nolatency/results/m1_filter.json → other_systems（m1_filter_v1 "pooled counts, clean emission table, step cost 0.04"；m1_filter_v2/v3 "noisy emission (accounts for 10 % mishearing) +0.008; balancing train datasets equally … +0.009; lower step cost 0.04→0.01 +0.015"；m1_filter_v5lo note 的 w_oct 曲线 0.6→0.637、0.3→0.654/0.048、0.2→0.677/0.137、0.15→0.687/0.195、0→0.707/0.405）、what_failed（7 项，含 "temperature on current-note evidence (0.7 or 1.3 both worse)"）、headline 0.6872（v5）、rows（v5 double 0.195/0.185/0.121/0.186）；systems/m1_filter_v1.py → P（lam 1.0、kappa 0.5、w_step 0.04、w_big 0.4、w_par 0.6、w_mid 0.004、w_oct 0.6、w_unis 1.0）；systems/m1_filter_v2.py → lam 1.5；futures/v3_nolatency/systems/ 下 m1_filter_v1.py … m1_filter_v7.py 与 m1_filter_v5lo.py 共 8 个；lb:3、lb:80、lb:81 = m1_filter_v1 / v5 / v5lo 的笔记本记录（旧分数 0.6143 / 0.6872 / 0.6540；lb:80 note "w_oct 0.15 (doubling ~0.2)"）；nd/m1_filter_v5lo → double 0.0481/0.0406/0.0266/0.0344；runs_mini/v4_1101/state_R1.json → diagnosis.diagnostic_configurations_evaluated_on_val "about 40"；why_a → variants.E_lam1.0.rt.head.music_nd 0.6233（why_a、why_c、why_d、why_e 里不偷看答案的单个 HMM 变体都不超过它）；PROGRAM.md §4.2（+0.005） -->

## 5. 和谁比、差多少、为什么赢

音乐分的定义（$o_{ik}$ = 她第 $i$ 个真实音在第 $k$ 拍里唱了多久，只算第 4 拍起、有和弦标签的拍；$m_i$ 是她的真实音，$h_i$ 是 Duet 的音，不唱算 0；$z_k$ 是真和弦，$t$ 是主音）：

$$\text{music\_nd}=\frac{\sum_{i,k}o_{ik}\,\mathbb 1\big[\,|m_i-h_i|\bmod 12\in\{3,4,7,8,9\}\,\big]\,\mathbb 1\big[(h_i-t)\bmod 12\in\mathrm{CT}(z_k)\big]}{\sum_{i,k}o_{ik}}$$

| 系统 | 是什么 | 实时能用 | 总分 | bach | folk | pop | pop909 | 抄她率（四个里最大） | 比 m1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| third_below | 基线（baseline）：唱她下方的三度（按大调音阶） | 是 | 0.5054 | 0.5429 | 0.4903 | 0.5121 | 0.4763 | 0.013 | -0.1152 |
| m3_seq_v5 | 三个 GRU 平均猜和弦 + 它自己的选音器（会抄她） | 是 | 0.6104 | 0.6241 | 0.6805 | 0.5713 | 0.5655 | 0.184 | -0.0103 |
| **m1_filter_v5lo** | HMM 前向滤波 + 不抄她的选音器 | 是 | **0.6206** | **0.6655** | **0.6766** | **0.5601** | **0.5804** | 0.048 | 0 |
| 上限 1 | m1 的选音器 + 直接告诉它这一拍的真和弦 | 否 | 0.8403 | 0.8925 | 0.8899 | 0.7779 | 0.8009 | 0.028 | +0.2197 |
| 上限 2：m4_oracle | 开卷系统（oracle）：告诉真和弦，用另一套选音规则 | 否 | 0.8388 | 0.8871 | 0.8784 | 0.7889 | 0.8009 | 0.040 | +0.2182 |

换一组听错的随机种子（seed 2），m1 是 0.6208（bach 0.6637、folk 0.6735、pop 0.5628、pop909 0.5831），和 0.6206 几乎一样。

<!-- src: 定义 = futures/v3_nolatency/rescore_nd.py（iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]]；k ≥ bench3.J = 4；z[k] ≥ 0；headline_nd = 四个平均）；nd/third_below、nd/m3_seq_v5、nd/m1_filter_v5lo → music_nd 与 double（抄她率列取四个数据集的最大值：0.0135 bach、0.1844 bach、0.0481 bach）；上限 1 = diag_rt → datasets.*.music_nd.told、headline_mean_of_4.music_nd.told 0.8403194、datasets.*.double.told 最大 0.0283（bach）；上限 2 = futures/v3_nolatency/results/nd_m4_oracle.json（headline_nd 0.8388254；double 最大 0.0397 folk），也见 runs_mini/v4_1101/report/board.txt 第 55–56 行；third_below 的规则见 futures/v3_nolatency/systems/third_below.py；m4_oracle 的选音规则见 systems/m4_oracle.py docstring；seed 2 = runs_mini/v4_1101/seed2/m1_filter_v5lo.json → headline_nd 0.6207917；"比 m1" = 未四舍五入的总分相减（m3_seq_v5：0.6103568 - 0.6206376 = -0.0103） -->

为什么赢：
- 比 third_below 高 0.115：third_below 只保证和她协和，不管和弦。两者和她协和的时间差不多（m1 0.944–0.959，third_below 0.948–0.954），但 Duet 的音是真和弦内音的比例，m1 是 0.603–0.732，third_below 只有 0.498–0.559。m1 的线条也更平顺：大跳占 Duet 换音的 0.3–1.1 %，third_below 是 1.8–10.8 %（被打分程序标成"太跳"）。
- 比 m3_seq_v5 高 0.010：三个 GRU 的平均其实猜和弦更准（准确率 0.484，m1 是 0.434），但 m3_seq_v5 自己的选音器常唱她的同一个音或八度（抄她率 0.087–0.184，m1 只有 0.027–0.048），音乐分把这些算错。旧分数把抄她算对，那时 m3_seq_v5 是 0.720，m1 是 0.654。m1 也不是每个数据集都赢：bach +0.041、pop909 +0.015，folk -0.004、pop -0.011。
- 离上限还差 0.22，但同一个选音器一旦知道真和弦就到 0.8403，和 m4_oracle 的 0.8388 一样好：选音器不是瓶颈，猜和弦才是。

<!-- src: lb:169 与 lb:170 → rows.*.cons（m1 0.9586/0.9585/0.9444/0.9493；third_below 0.9501/0.9544/0.9498/0.9482）、rows.*.fit（m1 0.7149/0.7320/0.6034/0.6311；third_below 0.5591/0.5119/0.5341/0.4975）、rows.*.big（m1 0.0028/0.0095/0.0112/0.0072；third_below 0.0178/0.0985/0.0621/0.1076）、flags（third_below ["ROUGH"]，bench3.py："ROUGH flag when big > 10 %"）、headline 旧分数（lb:169 m1 0.6540；lb:171 m3_seq_v5 0.7199）；diag_rt → headline_mean_of_4.chord_acc.gru.all.acc 0.4841、chord_acc.rt.all.acc 0.4335；nd/m3_seq_v5 → double 0.1844/0.1177/0.0999/0.0865；数据集差值 = nd/m1_filter_v5lo - nd/m3_seq_v5（+0.0414、-0.0039、-0.0112、+0.0148）；systems/m3_seq_v4.py → V3（dbl 0.2，above 0.0）与 systems/m3_seq_v1.py → Duet.step（候选覆盖整个 ALTO，含她上面） -->

## 6. 哪里失败、为什么

m1 漏掉的时间去哪了（占她计分歌唱时间的比例；每段漏掉的时间按下表从上到下，归到第一个符合的原因）：

| 原因 | 平均 | bach | folk | pop | pop909 |
|---|---:|---:|---:|---:|---:|
| 她的音开始那一拍没有和弦标签 | 0.0005 | 0 | 0 | 0.0014 | 0.0005 |
| 她的音还没唱完，和弦就换了 | 0.026 | 0.024 | 0.002 | 0.042 | 0.034 |
| 她下面没有能得分的音（知道真和弦也没有） | 0.069 | 0.021 | 0.050 | 0.118 | 0.085 |
| 她这个音被听错了 | 0.064 | 0.065 | 0.066 | 0.058 | 0.067 |
| **首选和弦猜错** | **0.201** | **0.207** | **0.178** | **0.207** | **0.213** |
| 首选和弦对了，还是没得分 | 0.019 | 0.018 | 0.026 | 0.012 | 0.020 |
| 合计（= 1 - music_nd） | 0.379 | 0.335 | 0.323 | 0.440 | 0.420 |

"她下面没有能得分的音"里，平均 0.042 的时间（pop 是 0.090）唱到她上面就能得分；只唱在她下面是我们的设计取舍。

<!-- src: diag_rt → headline_mean_of_4.miss_share.rt 与 datasets.*.miss_share.rt；diag_champs → table_4dp.m1_filter_v5lo（同样的数，4 位小数）与 systems.m1_filter_v5lo.datasets.*.buckets_plus_nd_minus_1 ≈ 0（各项加 music_nd 等于 1）；归类顺序：runs_mini/v4_1101/tools/diag_rt.py docstring 与 one_song()；diag_rt → no_good_below_but_above_ok_share：headline 0.0422、pop 0.0899 -->

同一个选音器，只换"对和弦知道多少"：

| 和弦从哪来 | 实时能用 | 总分 | bach | folk | pop | pop909 |
|---|---|---:|---:|---:|---:|---:|
| m1 自己实时猜（就是 m1） | 是 | 0.6206 | 0.6655 | 0.6766 | 0.5601 | 0.5804 |
| 听完整首歌再回头猜（离线平滑，offline smoothing） | 否 | 0.6229 | 0.6306 | 0.7103 | 0.5511 | 0.5995 |
| 告诉它上一拍的真和弦（完美记忆） | 否（用了真标签） | 0.7026 | 0.6758 | 0.7743 | 0.6740 | 0.6863 |
| 告诉它这一拍的真和弦（上限 1） | 否 | 0.8403 | 0.8925 | 0.8899 | 0.7779 | 0.8009 |

和弦准确率（她每个音开始时首选和弦对不对）：m1 是 0.434（bach 0.471、folk 0.571、pop 0.357、pop909 0.334）；永远猜 I 和弦（I chord，以主音为底的三和弦）是 0.299；三个 GRU 的平均是 0.484。

<!-- src: diag_rt → datasets.*.music_nd.{rt, smooth, past, told} 与 headline_mean_of_4.music_nd（0.6206 / 0.6229 / 0.7026 / 0.8403）；diag_rt → headline_mean_of_4.chord_acc.rt.all.acc 0.4335、datasets.*.chord_acc.rt.all.acc（0.4712/0.5714/0.3571/0.3344）、headline_mean_of_4.chord_acc.gru.all.acc 0.4841；永远猜 I：extra → mean.share_I 0.2991（state_R1.json → diagnosis.onset_chord_accuracy.always_I 0.299） -->

为什么猜不准（自动迭代第 1 轮的分析）：
1. 锁定（lock-on）不上：同一个和弦里她的第 1、2、3、4 个及以后的音，m1 的准确率是 0.434、0.412、0.447、0.437，听得越多并没有越准；要是告诉它上一拍的真和弦，会升到 0.392、0.654、0.835、0.879。
2. 她的旋律本身说不清是哪个和弦：按"这段她唱的时间有多少落在它的三个音上"排名，真和弦单独第一的段只占 1 %（bach）、22 %（folk）、11 %（pop）、14 %（pop909），平均 2.0–3.1 个三和弦并列第一；就算知道这段从哪开始、把段里的音完美累积，准确率也只有 0.45、0.43、0.47、0.50（第 1、2、3、4+ 个音）。
3. 于是信念退回到"平时哪个和弦最常见"：m1 在 42–49 % 的音上猜 I，真实的 I 只占 21–43 %；同一个和弦里相邻两个音之间，首选和弦有 36–39 % 会变。
4. 问题不在看不到未来（离线平滑只到 0.6229），而在不知道和弦（知道上一拍的和弦就到 0.7026）。合并的转移表是个折中（val 里 bach 74 % 的拍换和弦，pop 只有 26 %），但按数据集分开的表、甚至告诉它是哪个数据集，也只有 0.619，是次要原因。

<!-- src: diag_rt → headline_mean_of_4.chord_acc.rt.note_{1,2,3,4}_in_chord.acc（0.4337/0.4123/0.4469/0.4370）与 chord_acc.past.note_*（0.3917/0.6537/0.8353/0.8791）；why_d → datasets.*.all_spans.unique_best（0.0126/0.2246/0.1099/0.1380）、all_spans.mean_tied（3.12/2.33/2.49/2.02）、perfect_lockon_ceiling_acc_by_note_index（四个数据集平均 0.446/0.430/0.473/0.501）、argmax_flip_rate_within_span.m1（0.360/0.387/0.366/0.394）；why_b → confusions_m1_rt.*.guess_share[0]（0.417/0.485/0.488/0.438）与 true_chord_share[0]（0.307/0.430/0.248/0.211）；extra → per_ds.*.chord_change_rate_per_beat（bach 0.738、pop 0.264）；why_a → variants.T_dsA_oracle.rt.head.music_nd 0.6187；汇总见 state_R1.json → diagnosis.why_no_lock_on -->

最难的数据集是 pop：音乐分最低（0.5601），连上限 1 也最低（0.7779）。一是"她下面没有能得分的音"最多（0.118）：她唱得越低，alto 范围（最低 53）里她下面能选的音就越少；二是 pop 的旋律更少落在和弦内音上：她真实的歌唱时间里只有 0.656 是真和弦的和弦内音（bach 是 0.916），所以更难从旋律猜和弦（准确率 0.357；pop909 更低，0.334）。

<!-- src: nd/m1_filter_v5lo → pop 0.5601；diag_rt → datasets.pop.music_nd.told 0.7779、datasets.pop.miss_share.rt.no_good_note_below 0.1184；why_b → evidence.pop.chord_tone_share.true_all 0.6560、evidence.bach.chord_tone_share.true_all 0.9164；diag_rt → datasets.pop.chord_acc.rt.all.acc 0.3571、datasets.pop909.chord_acc.rt.all.acc 0.3344；"唱得越低选择越少"是由候选范围 53 到 min(77, m - 1) 推出的机制，不是对 pop 音高的统计 -->

听错的代价：如果她每个音都被听对，m1 能到 0.668（现在 0.6206）；被听错的音上，和弦准确率只有 0.285（听对的音是 0.450）。

<!-- src: extra → mean.music_nd_clean_melody 0.66805；diag_rt → headline_mean_of_4.chord_acc.rt.misheard.acc 0.2848、heard_right.acc 0.4497 -->

耳朵的判断：盲听（blind listening）时 Vincent 没有做二选一，他说所有片段都合格，问题都出在 Duet 实时配和弦（猜和弦）的地方。

<!-- src: futures/v4_selfiter/listen/result.json → code null、vincent_said、reading -->

标签的限制：我们的 7 个和弦都是用大调音阶（major scale）搭出来的，所以在小调（minor key）歌里，那个向上倾向主音的升高第 7 音（导音，leading tone）会被算成错——这是标签的限制，不是我们调出来的。同样，七和弦（seventh chord）多出来的那个音也不算和弦内音。

<!-- src: futures/v4_selfiter/PROGRAM.md §4.4 "Known limit" 与 §5.4 原句；duet_beat/theory.py → MAJOR、CT（7 个大调三和弦） -->

## 7. 下一步

已经做了：r2_mix_v2 只把和弦信念换成 $b\propto b_{\text{HMM}}^{0.3}\,b_{\text{GRU}}^{0.7}$（m1 的实时 HMM 与 m3_seq_v5 三个 GRU 平均的对数线性混合（log-linear mix）），选音器 6 个权重一个不改，val 上 0.6444（比 m1 高 0.0238，seed 2 是 0.6470）；Vincent 通过它的测验后，它才成为周二的新模型。

<!-- src: runs_mini/v4_1101/state_R4.json → presentable_champion（headline_nd 0.644428、seed2 0.647007、vs_m1.headline 0.0238、tuesday_model）；futures/v3_nolatency/systems/r2_mix_v2.py → ALPHA = 0.7；nd/r2_mix_v2 → headline_nd 0.6444280；runs_mini/v4_1101/seed2/r2_mix_v2.json → headline_nd 0.6470068 -->

## 8. 教授最可能问的 10 个问题

1. **每个音 Duet 到底知道什么？怎么证明它没偷看未来？**
   答：它只知道到此刻为止她每个音的起始时间和听到的音高（每个音有 10 % 会听错）、之前每个音的长度、主音、拍号和拍子网格，没有和弦标签，也不知道当前这个音会唱多久。因果检查在 24 首 val 歌、每首 3 个切点上把切点之后的音高、时长和全部和弦标签随机打乱，72 次检查里切点之前的输出全都没变。
   <!-- src: futures/v3_nolatency/bench3.py docstring；duet_beat/bench.py → NOISE = 0.10；runs_mini/v4_1101/review/m1_filter_v5lo.json → a_causal（checks 72，mismatches []）；PROGRAM.md §4.1(a) -->

2. **你们的 HMM 里，隐藏状态、转移、发射分别是什么？**
   答：隐藏状态（hidden state）是这一拍的和弦（大调的 7 个三和弦之一），转移是"上一拍和弦变成这一拍和弦"的概率，按这一拍在小节里的位置分成 6 张表。发射是"在这个和弦下她每个音级唱了多久"的概率，实际用时给每个音级留 0.1/12 的余量来容忍听错、再乘证据权重 λ = 1.5，而且她正在唱的音在选音之前就算进去。
   <!-- src: systems/m1_filter_v5lo.py → S = 7、tkey()、Model.logB（noise 0.1）、P['lam'] = 1.5、step()（先 append 当前音再 belief(k)） -->

3. **模型怎么训练？目标函数是什么？**
   答：不用梯度下降，就在 4 个数据集共 9117 首带标签的训练歌上数次数（和弦怎么转、每个和弦下各音级唱了多久），加一点伪计数后除以总数，不加伪计数时这正好是加权最大似然的解。每首歌算 $N_{\max}/N_d$ 次（bach 60.4 次、pop 1 次），让 8149 首 pop 不淹没 135 首 bach，不做这个平衡的话是 0.6177。
   <!-- src: systems/m1_filter_v5lo.py → Model；训练歌数见第 4 节表（135 + 511 + 8149 + 322 = 9117）；why_a → variants.T_bal0.rt.head.music_nd 0.6177 -->

4. **10 % 听错怎么处理？影响多大？**
   答：猜和弦时，发射概率给每个音级都留了余量，一个听错的音不会把正确和弦的概率打成 0；选音时只能按听到的音找协和音，所以听错时可能和她的真实音不协和。如果每个音都听对，m1 能到 0.668（现在 0.6206），听错的音上和弦准确率只有 0.285，听对的是 0.450。
   <!-- src: systems/m1_filter_v5lo.py → Model.logB、step()（按听到的 midi 选）；extra → mean.music_nd_clean_melody 0.66805；diag_rt → headline_mean_of_4.chord_acc.rt.misheard.acc 0.2848、heard_right.acc 0.4497 -->

5. **选音器怎么选？6 个权重分别管什么？**
   答：候选是 53–77 之间、严格低于她听到的音、而且和它协和的音，得分 = 它是和弦内音的概率 - 八度 0.3 - 每移动一个半音 0.01 - 超过 7 个半音的大跳 0.4 - 离 65 每个半音 0.004 - 平行五度/八度 0.6。第 6 个权重"同度 1.0"永远用不上，因为 Duet 只唱在她下面；一个候选都没有就不唱（val 上 46310 个音里有 291 个）。
   <!-- src: systems/m1_filter_v5lo.py → P、step()；duet_beat/theory.py → ALTO、ALTO_MID 65、CONSONANT；runs_mini/v4_1101/review/m1_filter_v5lo.json → d_range.silent 291；diag_champs → systems.m1_filter_v5lo.notes_fed 46310 -->

6. **music_nd 是什么？为什么跟她唱同一个音或八度不算对？**
   答：它是她真实歌唱时间（第 4 拍起、有和弦标签的拍）里，Duet 的音同时满足"和她的真实音协和""是这一拍真和弦的和弦内音""不是她的同一个音或八度"三条的时间比例，四个数据集再取平均。跟她唱同一个音只是回声不是和声，旧分数把它算对会让爱抄的系统虚高：m3_seq_v5 的旧分数是 0.720，音乐分只有 0.6104。
   <!-- src: futures/v3_nolatency/rescore_nd.py；lb:171 → headline 0.7199；nd/m3_seq_v5 → headline_nd 0.6104 -->

7. **GRU 猜和弦更准，为什么 m1 还能赢 m3_seq_v5？**
   答：三个 GRU 的平均确实猜得更准（准确率 0.484 对 0.434），但 m3_seq_v5 自己的选音器常跟她唱同一个音或八度（抄她率最高 0.184，m1 最高 0.048），这些时间在音乐分里都算错。所以总分 m1 是 0.6206、m3_seq_v5 是 0.6104，m1 赢在 bach（+0.041）和 pop909（+0.015），在 folk（-0.004）和 pop（-0.011）上反而略输。
   <!-- src: diag_rt → headline_mean_of_4.chord_acc.gru.all.acc 0.4841、chord_acc.rt.all.acc 0.4335；nd/m3_seq_v5 与 nd/m1_filter_v5lo（music_nd、double） -->

8. **0.62 离上限多远？差在哪里？**
   答：同一个选音器直接告诉它这一拍的真和弦能到 0.8403（开卷系统 m4_oracle 是 0.8388），而 m1 漏掉的 0.379 的时间里有 0.201 是首选和弦猜错，所以主要差在实时猜和弦。就算告诉真和弦也还会漏掉约 0.160，主要是她下面没有能得分的音（0.069）、听错（0.051）和她的音没唱完和弦就换了（0.037）。
   <!-- src: diag_rt → headline_mean_of_4.music_nd.told 0.8403、miss_share.rt.wrong_chord_guess 0.2011、miss_share.told（no_good_note_below 0.0687、misheard 0.0510、held_into_new_chord 0.0367）；futures/v3_nolatency/results/nd_m4_oracle.json → 0.8388；0.379 = 1 - 0.6206；0.160 = 1 - 0.8403 -->

9. **用了多少历史？历史有用吗？**
   答：滤波器把她过去所有的音压成一张 7 个和弦的概率表，但实际只记得大约 16 拍：只用当前这一拍是 0.6101，最近 4 拍 0.6203，最近 16 拍和整首歌都是 0.6206。历史总共只值 +0.0105，因为一个和弦下她的旋律常常同时符合 2 到 3 个三和弦，多听几个音也锁定不了（同一和弦里第 1、2、3、4+ 个音的准确率是 0.434、0.412、0.447、0.437）。
   <!-- src: runs_mini/v4_1101/report/history.txt（m1_filter_v5lo 段：0.6101 / 0.6203 / 0.6206 / 0.6206，history gain +0.0105；"the filter forgets notes older than about 16 beats on its own"）；why_d → all_spans.mean_tied；diag_rt → chord_acc.rt.note_*_in_chord.acc -->

10. **参数在哪里调的？会不会过拟合（overfitting）？**
    答：所有设置和版本选择都只在 val 上做，这个包里没有 test 集，所以这些是"调过的 val 分数"，没有独立的测试分数。换一组听错的随机种子（seed 2）重算是 0.6208（原来 0.6206），说明分数对听错的抽样很稳，但 val 本身被调过参带来的偏高我们没法量化。
    <!-- src: PROGRAM.md §1.3；runs_mini/v4_1101/seed2/m1_filter_v5lo.json → headline_nd 0.6207917 -->

## 9. 术语表

- 验证集（validation set, val）：用来调参和比较版本的歌；生活例子：模拟考。
- 测试集（test set）：最后才看、从不拿来调参的歌（这个包没有）；生活例子：正式高考。
- 训练集（training set）：拿来数次数、学规律的歌；生活例子：课本上的例题。
- 调参（parameter tuning）：反复改模型里的几个数，挑分数最高的；生活例子：拧收音机旋钮直到声音最清楚。
- 总分（headline）：四个数据集音乐分的平均；生活例子：四门课的平均分。
- 数据集（dataset）：同一来源的一批歌；生活例子：一本习题册。
- 众赞歌（chorale）：巴赫写的四声部教堂赞美诗；生活例子：教堂里大家一起唱的诗歌。
- 实时（real-time）：边听边决定，不等整首歌结束；生活例子：同声传译。
- 和声（harmony）：几个音同时响的效果；生活例子：朋友在你下面唱另一条线。
- 隐马尔可夫模型（Hidden Markov Model, HMM）：看不见的状态按概率一步步变，每一步产生看得见的东西；生活例子：在屋里看不到天气，只能从路人带没带伞猜有没有下雨。
- 隐藏状态（hidden state）：模型要猜但看不见的东西，这里是这一拍的和弦；生活例子：屋里的你看不到的天气。
- 前向滤波（forward filter）：每来一点新线索，就只用过去和现在更新"现在是哪个状态"的概率；生活例子：边看球赛边更新"谁会赢"。
- 和弦（chord）：几个一起听很和谐的音；生活例子：吉他一次扫出来的声音。
- 三和弦（triad）：隔一个音叠起来的三个音；生活例子：钢琴上 do、mi、sol 一起按。
- 大调（major key）：以大调音阶为基础、听起来明亮的调；生活例子：多数儿歌的明亮感。
- 大调音阶（major scale）：do re mi fa sol la si 七个音；生活例子：从 C 开始只弹白键弹一圈。
- 小调（minor key）：听起来偏暗、偏伤感的调；生活例子：电影悲伤场景的配乐。
- 调（key）：一首歌用哪组音、以哪个音为"家"；生活例子：同一首歌男生女生用不同的调唱。
- 主音（tonic）：一个调的"家"；生活例子：歌最后让人觉得"唱完了"的那个音。
- I 和弦（I chord）：以主音为底的三和弦；生活例子：歌曲结尾"回到家"的那个和弦。
- 七和弦（seventh chord）：三和弦上再叠一个音的四音和弦；生活例子：爵士乐里"带点味道"的和弦。
- 和弦内音（chord tone）：属于当前和弦的音；生活例子：扫 C 和弦时真正响的那几个音。
- 协和音程（consonant interval）：两个音一起响听着舒服的距离，这里是相差 3、4、7、8、9 个半音（再隔几个八度也算）或正好隔几个八度；生活例子：合唱里不刺耳的两个音。
- 三度（third）：相差 3 或 4 个半音，如 do 到 mi；生活例子：两人合唱最常用的和音距离。
- 八度（octave）：名字一样、一高一低的两个音；生活例子：男女"齐唱"其实差一个八度。
- 同度（unison）：完全同一个音；生活例子：两人唱一模一样的音高。
- 半音（semitone）：钢琴上相邻两个键（含黑键）的距离；生活例子：紧挨着的白键和黑键。
- 大跳（leap）：一次跳超过 7 个半音；生活例子：说话时突然从很低的声音跳到尖叫。
- 平行五度/八度（parallel fifths/octaves）：两个声部同方向走，前后都是五度或都是八度；生活例子：两人像影子一样同步，听起来像一条线。
- 声部进行（voice leading）：一条声部从一个音走到下一个音的方式，要平顺、少跳；生活例子：下楼梯一步一阶，不是一下跳五阶。
- 惩罚（penalty）：不想要的做法在得分里扣分；生活例子：打球犯规扣分。
- 选音器（chooser）：知道和弦信念后，给每个候选音打分、挑最高的规则；生活例子：知道大概会下雨后挑穿哪件外套。
- 中音声部（alto）：合唱里在女高音下面的声部，这里是 MIDI 53–77；生活例子：合唱团里比领唱唱得低的那一排。
- 旋律（melody）：一个接一个唱出来的主调子；生活例子：洗澡时哼的调。
- 拍（beat）：音乐均匀的脉搏；生活例子：听歌时点头的那一下。
- 小节（bar）：几拍为一组，第一拍最重；生活例子：华尔兹"嘭嚓嚓"三拍一组。
- 强拍（downbeat）：小节的第 1 拍；生活例子：数"一二三四"时的"一"。
- 拍号（meter）：每小节几拍；生活例子：进行曲"一二、一二"，华尔兹"嘭嚓嚓"。
- 拍子网格（beat grid）：每一拍什么时候来、在小节里排第几；生活例子：节拍器的滴答和重音。
- 起始时间（onset）：一个音开始的时刻；生活例子：按下琴键的那一刻。
- MIDI 音高（MIDI note number）：用整数表示音高，每加 1 高一个半音；生活例子：给钢琴键从左到右编号。
- 音级（pitch class）：不管高低八度，只看是 12 个音名里的哪一个；生活例子：高音 do 和低音 do 是同一个音级。
- 听错（mishearing）：Duet 听到的音和她真唱的不一样，这里每个音 10 % 的机会；生活例子：吵闹的餐厅里听错一个字。
- 随机种子（random seed）：决定"随机"结果的起点，种子一样结果就一样；生活例子：约定好的洗牌方法。
- 标签（label）：每一拍标好的"正确和弦"；生活例子：练习册后面的标准答案。
- 零延迟（zero latency）：她的音一开始，Duet 的音同时出来；生活例子：两个人同时开口，而不是一个等另一个。
- 因果检查（causality probe）：把未来打乱，看过去的输出变不变；生活例子：换掉试卷后面的题，前面写好的答案不该变。
- 梯度下降（gradient descent）：神经网络常用的一小步一小步改数字的训练法（m1 不用）；生活例子：蒙着眼顺着坡往下走找谷底。
- 转移表（transition table）：上一拍和弦变成这一拍和弦的概率表；生活例子：今天下雨、明天还下雨的可能性。
- 发射概率（emission probability）：某个和弦下她唱出某个音的可能性；生活例子：下雨天路人带伞的可能性。
- 训练目标（training objective）：训练时要让它最大（或最小）的那个量；生活例子：减肥时盯着的体重数字。
- 最大似然（maximum likelihood）：选让训练数据"最可能出现"的数字，这里就是次数除以总数；生活例子：抛很多次硬币，用正面出现的比例当正面的概率。
- 伪计数（pseudo-count）：没见过的情况也先算一点点次数，免得概率为 0；生活例子：新餐厅还没人评价，也先给个默认分。
- 数据集平衡（dataset balancing）：让每个数据集在训练里一样重；生活例子：投票按小组算，人多的组不能一手遮天。
- 信念（belief）：此刻对 7 个和弦各有多大把握的概率表；生活例子：天气预报说"很可能下雨"。
- 首选和弦（top guess）：信念里概率最大的那个和弦；生活例子：预报里可能性最大的那种天气。
- 证据权重（evidence weight, λ）：她唱的音在信念里被放大几倍（这里 1.5）；生活例子：更相信亲眼看到的，而不是平时的习惯。
- 准确率（accuracy）：猜对的比例；生活例子：考试里答对的题占多少。
- 音乐分（music_nd）：她歌唱时间里，Duet 同时"和她协和、是真和弦的和弦内音、又不是她的音或八度"的比例；生活例子：伴唱时既不刺耳、又贴合伴奏、又不是简单跟唱的时间占多少。
- 旧分数（raw music）：以前的分数，抄她也算对；生活例子：考试抄同桌也给分。
- 抄她（doubling）：Duet 唱她的同一个音或差几个八度的同名音；生活例子：回声，不是和声。
- 抄她率（double）：她歌唱时间里 Duet 在抄她的比例；生活例子：一节课里只是跟着老师复读的时间占多少。
- 基线（baseline）：拿来比的最简单方法；生活例子：不复习直接考的分数。
- 上限（ceiling）：告诉它真和弦后能到的分数；生活例子：开卷考试的分数。
- 开卷系统（oracle）：能看到真答案（真和弦）的系统，只用来量上限；生活例子：拿着答案对答案。
- 门控循环单元（gated recurrent unit, GRU）：一种按顺序读输入、带记忆的神经网络；生活例子：边读小说边记住前面情节的读者。
- 对数线性混合（log-linear mix）：两个概率表各取一个次方再相乘、缩放到总和为 1；生活例子：两位专家的意见按事先定好的分量综合。
- 离线平滑（offline smoothing）：听完整首歌再回头猜每一拍的和弦（不能实时用）；生活例子：赛后看录像复盘。
- 锁定（lock-on）：同一个和弦持续时，听到的音越多就越确定；生活例子：猜谜时提示越多越有把握。
- 保留规则（keep rule）：新系统要换掉旧的必须满足的条件，例如总分至少 +0.005；生活例子：新纪录要明显超过旧纪录才算数。
- 自动迭代（self-iteration）：在 Mac mini 上无人值守地一轮轮试新想法、打分、按规则保留或放弃；生活例子：厨师每晚试几道新菜，只留下明显更好吃的。
- 盲听（blind listening）：不告诉听的人是哪个系统，只凭耳朵评；生活例子：蒙眼试喝两杯咖啡。
- 过拟合（overfitting）：在调参用的数据上分数好，换新数据就变差；生活例子：只背了模拟考答案，正式考试就不会。

<!-- src（术语表里的数字）: duet_beat/theory.py → ALTO (53, 77)、CONSONANT {0, 3, 4, 7, 8, 9}、PC_NAMES（12 个音级）、N_CHORDS = 7；duet_beat/bench.py → NOISE = 0.10；systems/m1_filter_v5lo.py → P['lam'] 1.5、w_big 阈值 d > 7；PROGRAM.md §4.2 → +0.005 -->
