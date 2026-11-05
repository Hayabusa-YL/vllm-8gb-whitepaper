# Issue #65401 复现与修复记录（get_top_k_tokens 顺序 bug）

日期：2026-11-05
环境：vLLM 0.33.0（venv /opt/vllm-venv），RTX 4060 Laptop 8GB，WSL2
上游 issue：https://github.com/vllm-project/vllm/issues/65401
（2026-11-03 创建，open，已被 seansie0830 认领修复，PR 未提）

## 现象

`LogitsProcessor.get_top_k_tokens(..., return_log_probs=False)` 对选中的 k 个
logits 先乘 `scale` 再套 `soft_cap`；而 `forward()` 是全词表先 `soft_cap` 再乘
`scale`。当 scale != 1.0 且 soft_cap > 0 时两者不等价，返回的 top-k 分数与
完整 logits 不一致。

实测（vLLM 0.33.0，与 issue 环境一致，复现成功）：

- ids: [[0, 1]]
- top-k values: [2.9711713790893555, 1.7483489513397217]
- full logits 同 ids: [5.220370292663574, 1.9290765523910522]
- assert_close 失败，最大绝对差 2.249

注意：候选 id 选择本身正确（单调变换保持排序），问题只是返回的分数值。

## 根因

`vllm/model_executor/layers/logits_processor.py`，`get_top_k_tokens` 尾部：

    if self.scale != 1.0:
        values = values * self.scale
    if self.soft_cap is not None:
        values = torch.tanh(values / self.soft_cap) * self.soft_cap
    return ids, values

应为 soft_cap 先、scale 后（与 `forward()` 及 `return_log_probs=True`
分支 L282-285 一致）。

## 修复

交换两段顺序（2 行顺序调整，无逻辑变更）：

    if self.soft_cap is not None:
        values = torch.tanh(values / self.soft_cap) * self.soft_cap
    if self.scale != 1.0:
        values = values * self.scale
    return ids, values

本地已打补丁：`/opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`
原文件备份：`/tmp/logits_processor.py.bak`

## 验证

复跑 issue 提供的纯 CPU reproducer（无需下载模型）：

- 修复后 top-k values: [5.220370292663574, 1.9290765523910522]
- full logits 同 ids: [5.220370292663574, 1.9290765523910522]
- assert_close PASS，完全一致

## 上游状态

- 已有人认领（seansie0830 评论：swap order + 补单测，PR 将提）
- 本记录用于白皮书"issue 复现与验证"案例，不作为独立贡献
- 若上游修复落地，本地补丁可还原（备份在 /tmp）
