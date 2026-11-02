<!-- Phase D 草稿（report writer）。src 路径约定：RUN/ = /Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/ ；V3/ = /Users/san/claude_sandbox/music/duet_v0/futures/v3_nolatency/ ；SI/ = /Users/san/claude_sandbox/music/duet_v0/futures/v4_selfiter/ ；DB/ = /Users/san/claude_sandbox/music/duet_v0/duet_beat/ ；DER1、DER2 = RUN/report_drafts/tools/derived_numbers_v1.json、derived_numbers_v2.json（只把结果文件里的数相减、求 4 个数据集的平均；脚本 derive_numbers*.py 在同一目录）。 -->
# 诊断：Duet 的实时配和弦错在哪里，修好了哪一部分

> 你 2026-11-01 的问题：「关键的问题其实发生在duet，即那个配和弦的地方（实时）」<!-- src: SI/listen/result.json vincent_said -->

**三句话结论。** ① 你说得对：问题几乎全在实时（real-time）猜和弦——选音器（chooser）不变，只要告诉它她起音这一拍的真和弦，公平分（music_nd）就从 0.621 升到 0.840。② 根源是她在一个和弦下唱的几个音通常同时符合 2–3 个三和弦（triad），光凭音高分不出，所以 m1 的 HMM 只猜对 43%，而且多听几个音也不变准。③ 我们补上的是"更会记旋律的猜测者 + 不那么武断的信念"：准确率 0.434 → 0.491，分数 0.621 → 0.644（可上台的 r2_mix_v2）/ 0.653（不可上台的 r3_style_mix_v3），离"知道和弦"还差 0.19–0.20。<!-- src: RUN/diag/diag_rt_v1.json headline_mean_of_4 (music_nd rt 0.6206, told 0.8403; chord_acc rt 0.4335, note_1..4 0.434/0.412/0.447/0.437); RUN/diag/why/why_d_v1.json all_spans.mean_tied 2.02–3.12; RUN/diag/diag_champs_v1.json table_4dp acc 0.4913; RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json; RUN/nd/r3_style_mix_v3/results/nd_r3_style_mix_v3.json; DER2 remaining_gap_told_minus (0.1959 / 0.1871) -->

## 0. 名词
- 和弦信念（chord belief）$b(z)$：她每个音起音时，Duet 认为这一拍是 7 个三和弦 $z\in\{\mathrm{I,ii,iii,IV,V,vi,vii^{\circ}}\}$ 中哪一个的概率。隐马尔可夫模型（hidden Markov model, HMM）和门控循环网络（gated recurrent unit, GRU）都只负责给出 $b$。<!-- src: DB/theory.py CHORDS -->
- 选音器：在她下方、与她（听到的）音协和的音 $h$ 中取 $\arg\max_h\ \sum_z b(z)\,\mathbb 1[h\in\mathrm{CT}(z)]$ 减去声部进行代价（voice-leading cost）。第一项是"$h$ 是和弦音（chord tone）的概率"；代价项（唱她的八度、步距、大跳、离音域中心、平行五/八度）本轮一个没动。<!-- src: V3/systems/m1_filter_v5lo.py Filter.step, P (w_oct 0.3, w_step 0.01, w_big 0.4, w_mid 0.004, w_par 0.6) -->
- 公平分 music_nd：她唱歌的时间里（第 4 拍起），Duet 的音同时与她的真实音协和、是真实和弦的音、不是她的同音或八度的比例。下文"平均"= bach、folk、pop、pop909 四个数据集的平均。<!-- src: V3/rescore_nd.py; V3/bench3.py J = 4 -->

## 1. 同一个选音器，不同的和弦知识
<!-- src（下表）：RUN/diag/diag_rt_v1.json headline_mean_of_4.music_nd（rt / smooth / past / told） -->
| 选音器拿到的和弦知识 | music_nd（平均） |
|---|---:|
| m1 的实时 HMM（现在） | 0.6206 |
| 离线（offline）HMM：看完整首、包括未来，非因果（not causal） | 0.6229 |
| 告诉它上一拍的真和弦 | 0.7026 |
| 告诉它她起音这一拍的真和弦 | 0.8403 |

看到未来也只到 0.623：吃亏的不是"实时"，而是 HMM 对和弦的认识不够。知道上一拍就到 0.703：最值钱的是"现在在哪个和弦、刚才换没换"。0.6206 → 0.8403 是猜和弦的全部代价（m4_oracle 用自己的选音器也是 0.8388）；r2_mix_v2 补上了其中 10.8%，r3_style_mix_v3 补上 14.8%。<!-- src: V3/results/nd_m4_oracle.json; DER2 share_of_gap_closed_r2_mix_v2 0.108, share_of_gap_closed_r3_style_mix_v3 0.148 -->

## 2. m1 的失分桶（miss buckets，占她唱歌时间，平均）
<!-- src（下表）：RUN/diag/diag_rt_v1.json headline_mean_of_4.miss_share.rt 与 .told；每列加和 = 1（DER1 m1_miss_share_sum_plus_nd = 1.0） -->
| 失分原因 | m1 实时 | 告诉起音和弦后 |
|---|---:|---:|
| 和弦猜错（wrong chord guess） | 0.201 | 0 |
| 她下方没有合适的音 | 0.069 | 0.069 |
| 把她的音听错（misheard） | 0.064 | 0.051 |
| 她的长音拖进新和弦（held into new chord） | 0.026 | 0.037 |
| 和弦猜对仍没得分 | 0.019 | 0.003 |
| **得分 music_nd** | **0.621** | **0.840** |

另有 0.0005 是起音拍没有标签，每列加起来是 1。"和弦猜错"远大于其余各项，两个冠军把它降到 0.171（r2_mix_v2）/ 0.169（r3_style_mix_v3）。"下方没有合适的音"里有 0.042 其实在她上方有合适的音，但 Duet 按设计只唱她下方。<!-- src: RUN/diag/diag_rt_v1.json miss_share.rt.onset_beat_unlabelled 0.0005, no_good_below_but_above_ok_share 0.0422; RUN/diag/diag_champs_v1.json table_4dp headline.wrong_chord_guess (0.2011 / 0.1714 / 0.1685) -->

## 3. 起音时的和弦准确率（top-1 chord accuracy）
<!-- src（下表）：always-I = RUN/diag/recheck/recheck_rt_extra_result_20261101_145317.json share_I（DER1 always_I_*）；m1 实时、离线、3 个 GRU、告诉上一拍 = RUN/diag/diag_rt_v1.json chord_acc 的 rt、smooth、gru、past（DER1 top1_acc_*）；两个冠军 = RUN/diag/diag_champs_v1.json table_4dp acc -->
| 猜法 | 平均 | bach | folk | pop | pop909 |
|---|---:|---:|---:|---:|---:|
| 永远猜 I（always-I） | 0.299 | 0.307 | 0.430 | 0.248 | 0.211 |
| m1 实时 HMM | 0.434 | 0.471 | 0.571 | 0.357 | 0.334 |
| 离线 HMM（看完整首） | 0.433 | 0.382 | 0.623 | 0.352 | 0.374 |
| 3 个 GRU 的平均 | 0.484 | 0.530 | 0.654 | 0.366 | 0.386 |
| r2_mix_v2（可上台） | 0.491 | 0.539 | 0.653 | 0.370 | 0.403 |
| r3_style_mix_v3（最高分） | 0.492 | 0.533 | 0.650 | 0.377 | 0.406 |
| 告诉它上一拍的真和弦 | 0.640 | 0.401 | 0.751 | 0.738 | 0.671 |

HMM 0.434，"永远猜 I"也有 0.299；离线 HMM 看完整首仍是 0.433。pop 最难，GRU 在 pop 上几乎没进步。bach 在 74% 的拍子边界上换和弦，所以"知道上一拍"对它帮助最小。<!-- src: RUN/r4/A/calib_20261101_160528.json calibration_val.bach.all.actual_change 0.7369 -->

## 4. 锁定曲线（lock-on curve）和"只靠音高"的上限
"锁定"= 同一个和弦段（span）里听到的音越多，就猜得越准。
<!-- src（下表）：m1、3 个 GRU、告诉上一拍 = RUN/diag/diag_rt_v1.json chord_acc.<rt|gru|past>.note_1..4_in_chord（DER1 lockon_mean4）；两个冠军 = RUN/diag/diag_champs_v1.json table_4dp n1..n4+；上限 = RUN/diag/why/why_d_v1.json perfect_lockon_ceiling_acc_by_note_index 的 4 数据集平均（DER1 why_d） -->
| 同一和弦里的第几个音 | 1 | 2 | 3 | ≥4 |
|---|---:|---:|---:|---:|
| m1 实时 HMM | 0.434 | 0.412 | 0.447 | 0.437 |
| 3 个 GRU 的平均 | 0.457 | 0.468 | 0.489 | 0.518 |
| r2_mix_v2 | 0.468 | 0.477 | 0.511 | 0.521 |
| r3_style_mix_v3 | 0.464 | 0.482 | 0.512 | 0.538 |
| 上限：已知和弦从哪拍开始，只用 m1 的音高证据 | 0.446 | 0.430 | 0.473 | 0.501 |
| 告诉它上一拍的真和弦 | 0.392 | 0.654 | 0.835 | 0.879 |

m1 是平的：同一个和弦里多听三个音也不更准，从没"锁定"；"告诉上一拍"那一行才是锁定的样子。倒数第二行最关键：就算知道和弦从哪拍开始，只凭她的音高，上限也只有 0.43–0.50；冠军已经贴着、甚至略高于这条线，因为 GRU 还用了节奏、拍位和和弦进行（chord progression）。所以剩下的路不在"更仔细地听音高"，而在"更懂和弦什么时候换、换到哪"。

## 5. 为什么：她的旋律常常让 2–3 个三和弦打平
- 标注和弦是"唯一覆盖她最多音"的三和弦，只占 bach 1%、folk 22%、pop 11%、pop909 14% 的和弦段，平均有 3.1 / 2.3 / 2.5 / 2.0 个三和弦并列第一。例：C 大调里她唱 C、E（do、mi），I（C-E-G）和 vi（A-C-E）都含这两个音。<!-- src: RUN/diag/why/why_d_v1.json all_spans.unique_best (0.0126/0.2246/0.1099/0.1380), mean_tied (3.12/2.33/2.49/2.02) -->
- m1 的滤波（filter）：
$$b_k(z)\ \propto\ \Big[\textstyle\sum_{z'} b_{k-1}(z')\,A_{\tau(k)}(z',z)\Big]\cdot\textstyle\prod_{p}\tilde B(p\mid z)^{\lambda\,d_k(p)},\qquad \tilde B=0.9\,B+\tfrac{0.1}{12},\ \ \lambda=1.5$$
前一项是按拍位 $\tau(k)$ 的转移（transition），后一项是她这一拍的音高证据（emission），$d_k(p)$ 是音级 $p$ 在这拍响了多久，$\tilde B$ 已算进 10% 听错（mishearing）。几个 $z$ 的证据打平时，$b_k$ 只由转移和先验（prior）决定。<!-- src: V3/systems/m1_filter_v5lo.py Model (logB, noise 0.1), Filter._beat_ll / _advance (lam 1.5); DB/bench.py NOISE = 0.10 -->
- 结果：m1 有 42–49% 的音猜 I（真实只有 21–43% 是 I）；同一和弦段里相邻两个音的猜测有 36–39% 会换来换去。<!-- src: RUN/diag/why/why_b_v1.json confusions_m1_rt guess_share[I] 0.417/0.485/0.488/0.438, true_chord_share[I] 0.307/0.43/0.248/0.211; RUN/diag/why/why_d_v1.json argmax_flip_rate_within_span.m1 0.360/0.387/0.366/0.394 -->

## 6. 修好了什么
1. **更好的猜测者（E1）**：GRU 读她从头到现在的音高、时值、拍位，训练目标就是"这个音覆盖的各拍的和弦"。准确率 0.434 → 0.484、分数 0.6440，但它太自信（平均最高概率 0.528，HMM 0.451），换和弦时更常大跳（big leap）：folk 0.0212，超过上限 0.0195。<!-- src: V3/systems/m3_seq_v1.py (features, soft target); DER1 top1_acc_mean4; RUN/r3/C/diag_20261101_153920.json mean4.mean_top_prob (0.5279 / 0.4508); DER2 ideas.E1 r1_gru_chooser_v1 (0.644003, big_folk 0.0212), keep_rule_folk_big_limit_vs_m1 0.0195 -->
2. **对数线性混合（log-linear mix，E2 = r2_mix_v2）** $b\propto b_{\mathrm{HMM}}^{0.3}\,b_{\mathrm{GRU}}^{0.7}$：准确率 0.491，平均最高概率 0.504，folk 大跳 0.0185（合规），分数 0.6444。增益来自 GRU 的长记忆：只看她当前一个音时与 HMM 几乎一样（0.6114 对 0.6101）；看最近 4 拍 / 16 拍 / 整首时，比 HMM 高 0.0141 / 0.0216 / 0.0238。<!-- src: V3/systems/r2_mix_v2.py; RUN/diag/diag_champs_v1.json table_4dp r2_mix_v2 acc 0.4913; RUN/r3/C/diag_20261101_153920.json mean_top_prob 0.5041; DER2 ideas.E2 r2_mix_v2 big_folk 0.0185; RUN/report/history.txt -->
3. **更平的信念 + pop 专用 HMM（E2×E3 = r3_style_mix_v3）**：准确率几乎不变（0.4913 → 0.4916），但平均最高概率 0.504 → 0.458。信念更平，选音器就更偏向几个可能和弦共有的音（她唱 E、信念在 I 和 vi 之间各半时，在她下面唱 C 两边都得分），也更少唱她的同音/八度（bach double 0.055 → 0.035）；pop 专用 HMM 让 pop 准确率 0.370 → 0.377。消融（ablation，都取 $\alpha=0.5$）：通用 HMM $\lambda=1.5$ → 通用 $\lambda=1.0$ → 风格 HMM：0.6428 → 0.6497 → 0.6532。<!-- src: RUN/diag/diag_champs_v1.json table_4dp acc (0.4913 / 0.4916), pop acc (0.3697 / 0.377); RUN/r3/C/diag_20261101_153920.json mean_top_prob (0.5041 / 0.4584); RUN/nd/r2_mix_v2 与 RUN/nd/r3_style_mix_v3 的 nd 文件 bach double (0.0553 / 0.0351); RUN/r3/C/ablate_20261101_154122.json (0.642759 / 0.649741 / 0.65323); RUN/state_R3.json champion.why_it_wins -->

## 7. 没用的，以及为什么
- **改转移表**（在 m1 上）：更"粘"（sticky，留在原和弦的概率 ≥ 0.70 / 0.85 / 0.95）0.6193 / 0.6114 / 0.6062；不做语料平衡 0.6177；按语料分表、且告诉它是哪个语料 0.6187；按拍子的和弦时长先验（E3 原样）0.6139 / 0.6212；自适应表 0.6114 / 0.6216：都没有明显超过 m1 的 0.6206。瓶颈不是"多久换一次"，而是"打平时是哪一个"；更粘只会让错的猜测停得更久。<!-- src: DER1 m1_tweaks_rt_music_nd (why_a T_sticky70/85/95, T_bal0, T_dsA_oracle; why_e DUR_pooled_lam1.5/1.0; why_b MIX4_tables_lam1.5/1.0); RUN/diag/why/why_a_v1.json pstay_tables -->
- **E4**（按她的音会盖住的几拍算契合）：R3 最好 +0.0027（多半是少唱同音），R4 −0.0003。和弦在她长音底下换掉的时间只占 folk 0.4%、bach 4.8%、pop 8.7%、pop909 7.9%，folk 要的 +0.003 连完美预测都拿不到；GRU 的训练目标本来就覆盖这些拍。<!-- src: DER2 ideas.E4_R3 r3_cover_fit_v3 (+0.0027), ideas.E4_R4 (−0.0003); RUN/state_R3.json ideas E4; RUN/r3/A/where_20261101_154310.json time_share.later_chord_changed (DER1 r3A_time_share_later_chord_changed); RUN/state_R4.json ideas_round_4 E4.why -->
- **E5**（按 10% 听错，挑与她**真实**音期望协和的音）与"八二分账"：听错让 r3_style_mix_v3 每个数据集掉 0.041–0.056；把她的真实音喂给信念只找回其中 13–24%，约八成损失在"选音"上，因为 Duet 只能与它听到的音协和。听错的音有 53–55% 仍在大调音阶内，单看一个音认不出；信念即使用她的真实音，准确率也只从 0.492 升到 0.517。在音阶外听错音上赢的分，又在她真唱的半音（chromatic）上输回去：R3 −0.0008 ~ +0.0005，R4 −0.0005 / −0.0042（bach −0.017）。<!-- src: RUN/r4/B/check_why_20261101_160400.json (cost_total 0.0412–0.0565, cost_belief / cost_total 0.134–0.237, misheard in-scale share 0.530–0.550 = DER1 r4B_mishearing); RUN/r4/B/check_belief_20261101_160733.json (DER1 r4B_top1_true_notes_vs_heard 0.4916 → 0.5173); RUN/r3/B/check_where_20261101_154642.json; DER2 ideas.E5_R3, ideas.E5_R4 (r4_true_cons_v2 bach −0.0171) -->

## 8. 最好的下一步：持音边界（held-note boundary）
- 发现：她的一个音拖过拍子边界时，标注和弦很少换：folk 3.6%（m1 的通用转移表假设 22%），pop 19%（表 32%），pop909 28%（表 34%）；bach 在持音边界上换 26%，在全部边界上换 74%（表 38%）。<!-- src: RUN/r4/A/calib_20261101_160528.json calibration_val（held / all 的 actual_change 与 implied_change[0] = own_pooled；DER1 r4A_calibration_val） -->
- 含义与做法：现在的 HMM 每拍都按同一张拍位表转移，不管她的音是否还拖着，信念被无谓地摊平。下一步让转移多一个条件，$A_{\tau(k),\,\text{held}}$ 与 $A_{\tau(k),\,\text{new}}$（她的音是否跨过这条边界），先看 folk、bach 的准确率和 music_nd。R4 只在 E4 的"预测会盖住的拍"里用过持音表（−0.0005 / −0.0008），还没放进 HMM 滤波本身；它不在 §4.4 里，所以本轮没做。<!-- src: RUN/r4/A/calib_20261101_160528.json configs held_beat / held_cover delta_headline (−0.000511 / −0.000799); RUN/state_R4.json ideas_round_4 E4.finding_for_later -->
- 已知限制（不调）：标签只有大调的 7 个三和弦，七和弦和小调里升高的导音（leading tone）都算错。<!-- src: SI/PROGRAM.md §4.4; DB/theory.py CHORDS -->

全部数字是 val 上的，也在 val 上调（没有 test）。
