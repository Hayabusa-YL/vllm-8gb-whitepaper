# AWQ vs GPTQ 量化对比实测记录

日期：2026-11-05
环境：vLLM 0.33.0，RTX 4060 Laptop 8GB，WSL2，共同参数
（--gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager）

## 方法

OpenAI 兼容接口 + 多线程并发，统计 usage.completion_tokens 总和 ÷ 墙钟耗时；
prompt 固定"写一篇关于人工智能发展历史的文章"，temperature=0；
每场景多轮取稳定值（单请求有首次冷启动波动，取后轮）。

## 结果

| 场景 | AWQ 4-bit | GPTQ-Int4 |
|---|---|---|
| 权重体积 | 5.19GB | 5.2GB |
| GPU 显存占用 | 7.19 / 8.0 GiB | 7.19 / 8.0 GiB |
| KV cache | 0.98 GiB / 18,368 tok | 0.97 GiB / 18,240 tok |
| 1×256 | 53.2 tok/s | 48-54 tok/s（波动） |
| 8×256 | 411 tok/s | 406 tok/s |
| 8×512 | 370-405 tok/s | 401-412 tok/s |

## 结论

- 两者性能在测量噪声内（±5%），显存占用一致，KV 容量差 0.7%
- 8G 卡选哪款看生态即可，不是性能分水岭
- 注意：WSL2 切换服务时残留 EngineCore 进程会占显存导致新服务启动被拒
  （"Free memory ... less than desired utilization"），需 pkill -9 -f EngineCore
- 最终服务状态：AWQ 在 8000 端口

## 局限

- 单轮/双轮数据，未做统计检验；8×512 波动大（349-405），多跑可收窄
- bf16 因显存无法同机对比；GGUF 需 llama.cpp 栈，vLLM 0.33 无 GGUF 后端
