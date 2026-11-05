# Issue #64291 复现与修复记录（logprobs 校验缺口）

日期：2026-11-05
环境：vLLM 0.33.0（venv /opt/vllm-venv），RTX 4060 Laptop 8GB，WSL2
上游 issue：https://github.com/vllm-project/vllm/issues/64291
（2026-10-28 创建，open，0 评论，无人认领）

## 现象

当 `--max-logprobs` 为正整数且大于模型 vocab_size 时（默认 max_logprobs=20），
请求 `logprobs` 数量大于 vocab_size 的请求能通过校验，直到 worker 里
`compute_topk_scores()` 调 `torch.topk(logits, num_logprobs)` 才崩：

```
RuntimeError: selected index k out of range
```

预期是请求在校验期就被拒（4xx），或显式 clamp 到 vocab_size。

## 复现（vLLM 0.33.0，与 issue 一致）

mock model_config（max_logprobs=20，vocab_size=10）：

- `SamplingParams(logprobs=20)._validate_logprobs(model_config)` → 通过，未拒绝
- `compute_topk_scores(logits(2,10), num_logprobs=20)` → `RuntimeError: selected index k out of range`

复现脚本：/tmp/repro64291.py（纯 mock + 小 CUDA tensor，10 秒内跑完）

## 根因

`vllm/sampling_params.py` 的 `_validate_logprobs()` 开头：

    max_logprobs = model_config.max_logprobs
    if max_logprobs == -1:
        max_logprobs = model_config.get_vocab_size()

只对 -1 特判；正值 max_logprobs 大于 vocab_size 时不 clamp。后面
logprobs / prompt_logprobs / prompt_logprob_token_ids 宽度都拿它比较，全部放行。

## 修复

clamp 到 vocab_size：

    max_logprobs = model_config.max_logprobs
    vocab_size = model_config.get_vocab_size()
    if max_logprobs == -1:
        max_logprobs = vocab_size
    else:
        max_logprobs = min(max_logprobs, vocab_size)

本地已打补丁：`/opt/vllm-venv/lib/python3.12/site-packages/vllm/sampling_params.py`
原文件备份：`/tmp/sampling_params.py.bak`

## 验证

复跑同一 reproducer：

- validate: RAISED VLLMValidationError
  "Requested sample logprobs of 20, which is greater than max allowed: 10"
- worker 侧仍会崩（本修复只在校验侧，目的是合法请求不会到达 worker）

## 上游状态

- 无人认领、无 PR。修复是 3 行 clamp，可随时转成 PR（等阿叶定）。
- 本记录用于白皮书第三案例。
