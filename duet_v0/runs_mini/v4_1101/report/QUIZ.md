<!-- QUIZ.md（Phase D finalizer，2026-11-01）。src: 10 个问题 = report/CARD_new_zh.md 第 8 节（原样，不带答案）；5 个要背的数 = futures/v4_selfiter/PROGRAM.md §5.7（0.505、0.621、冠军、0.839、double 的意思），这里只给题目、不给数值，数值在 CARD_new_zh.md 文首；通过标准 = PROGRAM.md §5 与 runs_mini/v4_1101/state_R4.json presentable_champion.tuesday_model。 -->
# 测验：r2_mix_v2（Duet 新模型）

我在聊天里一题一题问；你用自己的话回答，我只指出漏掉的部分。全部是 val 上的数字，也在 val 上调（没有 test）。

## 10 个问题

1. 每听到她一个音，Duet 拿到什么、给出什么？
2. 和老模型 m1 比，到底改了什么？
3. music_nd 怎么算？为什么不算唱她的同音或八度？
4. GRU 读什么、学什么？
5. 为什么要把 HMM 和 GRU 相乘，不只用 GRU？
6. α = 0.7 怎么定的？会不会对 val 过拟合（overfitting）？
7. 为什么不直接用 GRU 自带的选音器（m3_seq_v5）？
8. 怎么保证它没偷看未来、没作弊？
9. 离"告诉你和弦"的天花板还差多少？差在哪？
10. 哪个数据集最难？为什么？

## 5 个要背的数（凭记忆回答）

1. 基线 third_below（唱她下方三度）的 music_nd 是多少？
2. 旧模型 m1_filter_v5lo 的 music_nd 是多少？
3. 可上台的冠军 r2_mix_v2 的 music_nd 是多少？
4. 告诉它真和弦时的天花板（m4_oracle）是多少？
5. double（抄她, doubling）是什么意思？

通过 = 10 题答对至少 8 题，且 5 个数字都在 ±0.01 以内；通过后 r2_mix_v2 才成为周二的模型，否则讲 m1_filter_v5lo（CARD_m1_zh.md）。
