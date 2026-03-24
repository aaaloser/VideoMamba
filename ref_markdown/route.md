先定义判据（每次跑完看 ptq_block_bits.json）

目标高位层比例建议在 25% 到 60%（32层里大约 8 到 19 层是8bit）

如果低于 10%，通常会接近全4bit，精度会明显掉

如果高于 90%，通常会接近全8bit，压缩收益太小

第一阶段：粗搜（9组）

固定 alpha=0.5, beta=0.25, gamma=0.25, num_groups=4

扫 tau_percentile: 75, 80, 85

扫 high_ratio_threshold: 0.12, 0.16, 0.20

固定 calib_batches=32, quick_batches=16

low_bit 先用 6（先找稳定区），high_bit=8

第二阶段：细搜（在第一阶段最好2组附近）

tau_percentile 以步长 2 微调（例如 78, 80, 82）

high_ratio_threshold 以步长 0.02 微调（例如 0.14, 0.16, 0.18）

把 low_bit 从 6 换成 4，再复测一次，看精度与压缩折中

第三阶段：结构超参数

num_groups 扫 2, 4, 8（会改变时空统计粒度）

alpha/beta/gamma 扫两组：

0.5/0.25/0.25（当前）

0.4/0.3/0.3（提高熵项权重，通常减少“全高或全低”极端）

你现在这类“又几乎全8”的直接调参方向

提高 high_ratio_threshold（例如 0.10 -> 0.16/0.18）

或提高 tau_percentile（例如 80 -> 85）

两者同时小步上调，优先让高位层比例回到 25% 到 60%