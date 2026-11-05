# 8GB 显存跑 7B 大模型：vLLM 消费级显卡极限实战白皮书

一张 RTX 4060 Laptop 8GB 显卡，把 Qwen2.5-7B（AWQ / GPTQ-Int4）在 vLLM 0.33.0 上跑通的完整记录：显存怎么分、吞吐到哪、三个真实上游 issue 怎么复现和修。

- 正文：[whitepaper.md](whitepaper.md)
- 版本：v0.3（2026-11-05）

## 里面有什么

| 章节 | 内容 |
|---|---|
| 第 1-2 章 | 为什么 8GB 能做、硬件软件基线 |
| 第 3 章 | WSL2 + vLLM 环境搭建踩坑实录 |
| 第 4 章 | 极限压测：显存拆解 7.3/8.0 GiB、吞吐 53 tok/s 单请求 / 414 tok/s 8 并发、4k-8k 长上下文 TTFT 曲线、AWQ vs GPTQ 对比、enforce-eager 开关实测 |
| 第 5 章 | 三个 issue 复现与修复：#49497 FlashInfer JIT 崩溃、#65401 采样分数一致性、#64291 logprobs 校验缺口 |
| 第 6 章 | 8GB 显存优化 SOP |
| 第 7 章 | 局限与可复现性 |

## 关键结论（先看这个）

1. 4-bit 量化后 Qwen2.5-7B 权重约 5.2 GiB，8GB 卡上留给 KV cache 的余量约 1 GiB，上下文上限就是 8192。
2. `--gpu-memory-utilization` 从 0.85 起，Windows 桌面占用下 0.9 会被拒（实测可用 6.88 / 8.0 GiB）。
3. `--enforce-eager` 在 8GB 上不是建议而是唯一可行：关掉后 CUDA Graph 捕获占 0.37 GiB，KV 余量变负、服务起不来。
4. AWQ 与 GPTQ-Int4 同参数吞吐差在测量噪声内（±5%），选哪个看生态，性能不是分水岭。

## 复现材料

- [audit-flashinfer-49497.md](audit-flashinfer-49497.md)：FlashInfer 采样器 JIT 崩溃的根因、补丁与验证
- [audit-get-top-k-tokens-65401.md](audit-get-top-k-tokens-65401.md)：`get_top_k_tokens` 分数顺序 bug 的纯 CPU 复现
- [audit-logprobs-64291.md](audit-logprobs-64291.md)：`--max-logprobs` 校验缺口
- [audit-awq-vs-gptq.md](audit-awq-vs-gptq.md)：AWQ / GPTQ 同机对比记录
- [repro_64291.py](repro_64291.py)：#64291 复现脚本
- [sampling_params-64291.patch](sampling_params-64291.patch)：#64291 本地补丁

## 复现命令

```bash
# 启动（8GB 显卡）
vllm serve <model> --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --port 8000 --host 0.0.0.0 --enforce-eager

# 单请求
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

吞吐统计口径：OpenAI 兼容 HTTP 接口 + 多线程并发，`usage.completion_tokens` 总和 ÷ 墙钟，三轮取稳定值，不含网络延迟。

## 分工说明

选题、硬件环境与验收由作者本人负责；实测执行、排障、数据分析与初稿撰写由本地 AI 智能体完成，所有数据来自本机真实运行日志。文中三个 issue 的本地补丁在上游均已有人认领修复，本文只做复现与验证记录，不重复提交 PR。

## 说明

- 数据仅覆盖 RTX 4060 Laptop 8GB + Qwen2.5-7B（AWQ/GPTQ），换卡换模型需重测。
- vLLM 迭代快，第 5 章的 issue 状态以 2026-11-05 为准。
