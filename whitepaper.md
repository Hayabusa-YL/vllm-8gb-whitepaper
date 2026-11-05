# 8GB 显存跑 7B 大模型：vLLM 消费级显卡极限实战白皮书

> 版本：v0.3（2026-11-05）
> 测试机：Windows 笔记本，RTX 4060 Laptop 8GB，24GB 内存
> 引擎：vLLM 0.33.0，模型 Qwen2.5-7B-Instruct-AWQ / GPTQ-Int4
> 目标读者：只有一张 8GB 显卡、想本地跑 7B 模型但被 OOM 和报错劝退的开发者

---

## 第 1 章 为什么盯上 8GB 这个数字

RTX 4060 Laptop 8GB 是 2024 年以来最普及的消费级独显之一。几乎所有"本地跑大模型"的教程都以 24GB 卡（4090）为基准写参数，8GB 用户照抄就是一路 OOM。但 8GB 并非跑不了 7B：**4-bit 量化后的 Qwen2.5-7B 权重约 5.2GB，剩下的 2.8GB 才决定你能不能跑、能跑多稳**。

这本白皮书只回答三个问题：

1. 8GB 显存到底能跑到什么程度（实测数据，不猜）；
2. 从零到跑通要踩哪些坑、怎么绕（vLLM + WSL2 环境实录）；
3. vLLM 社区里与低显存相关的真实 issue 长什么样、怎么定位和修复（附复现与补丁）。

所有数据来自同一台机器的真机实测，命令与日志原文可复现。

---

## 第 2 章 硬件与软件基线

### 2.1 硬件

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB GDDR6，3072 CUDA 核心，45W 功耗墙 |
| 驱动 | 595.80（Windows 宿主，WSL 直通） |
| 主机 | Windows 11 + WSL2（Ubuntu 24.04 LTS） |
| 内存 | 24GB DDR5 |

### 2.2 软件（截至 2026-11-05）

| 组件 | 版本 | 说明 |
|---|---|---|
| vLLM | 0.33.0 | PyPI 正式版，CUDA 12.8 构建 |
| PyTorch | 2.8.0+cu128 | vLLM 依赖自动安装 |
| FlashInfer | 0.7.2.post1 | JIT 采样后端 |
| flashinfer-cubin | 0.6.13（与 FlashInfer 不匹配，见第 5 章） | 预编译内核包 |
| Python | 3.12.3 | 系统 Python |
| 模型 | Qwen/Qwen2.5-7B-Instruct-AWQ | 4-bit AWQ，5.19GB |
| 模型 | Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 | 4-bit GPTQ，5.2GB |

> 基线原则：不替换、不升级系统组件。Python 依赖只装在 venv 内；驱动与 CUDA 不动。所有改动可回滚。

---

## 第 3 章 环境搭建实战（踩坑实录）

本节按真实操作顺序写。每一步都附踩过的坑。

### 3.1 WSL2 + GPU 直通

Windows 侧确认驱动支持 WSL GPU 直通后，安装 Ubuntu 24.04 的 WSL 发行版。关键验证：

```bash
nvidia-smi   # 能看到显卡，算直通成功
```

常见坑：nvidia-smi 找不到 → 检查 Windows 驱动版本与 WSL2 版本；WSL 内不要另装驱动。

### 3.2 安装 vLLM（最耗时的一步）

vLLM 0.33.0 从 PyPI 安装时要拉一堆 nvidia wheel（合计约 8GB），走境外源极慢。实测坑点：

1. **换国内镜像**：清华/阿里镜像同步正常，但 nvidia 系列 wheel 很大，镜像同步本身也有延迟。
2. **下载慢/超时**：vLLM 对 wheel 下载完整性敏感，中途断开会留下坏缓存，需清空 `~/.cache/pip` 重来。
3. **耗时**：网络正常时十几分钟起，镜像同步慢时可能接近半小时，期间别关终端。

```bash
python3 -m venv /opt/vllm-venv
/opt/vllm-venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.33.0
```

### 3.3 下模型

```bash
/opt/vllm-venv/bin/python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ'))"
```

AWQ 版权重 5.19GB。不要下 bf16 版（7B 权重约 15GB，8G 显卡放不下）。

### 3.4 第一次启动（必踩的坑）

直接 `vllm serve` 大概率不是 OOM 就是别的错。8G 卡必须显式限制：

```bash
vllm serve <模型路径> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enforce-eager
```

三个参数的实测理由（详见第 4 章）：

- `--gpu-memory-utilization 0.85`：0.9 以上在 Windows 桌面占用下会被拒绝启动（实测 0.92：启动时可用 6.88 / 8.0 GiB，低于请求的 7.36 GiB，进程直接退出）；
- `--max-model-len 8192`：KV cache 容量天花板；
- `--enforce-eager`：在这张 8G 卡上不是"建议"而是**唯一能起来的配置**——关掉后 CUDA Graph 捕获要占 0.37 GiB，KV cache 余量直接变成负数（实测见 4.3）。

### 3.5 启动成功的标志

看到 `Application startup complete.` 且 `/health` 返回 200 即可用。

---

## 第 4 章 极限压测：8GB 能跑到什么程度

### 4.1 显存占用拆解（8192 上下文，0.85 利用率）

| 项 | 大小 |
|---|---|
| 模型权重（AWQ 4-bit） | 5.29 GiB |
| 激活峰值 | 0.25 GiB |
| KV cache | 0.98 GiB（18,368 tokens） |
| **合计** | **约 7.3 / 8.0 GiB** |

结论：8G 卡跑 7B AWQ 后，留给 KV cache 的余量约 1GB，**上下文上限就是 8192 tokens**（实测 prompt 达到 8193 tokens 时被 400 拒绝，错误信息明确提示超过 max-model-len）。

### 4.2 吞吐实测（Qwen2.5-7B AWQ，8192 上下文）

测试方法：通过 OpenAI 兼容 HTTP 接口 + 多线程并发，统计 `usage.completion_tokens` 总和 ÷ 墙钟耗时；三轮取稳定值，不含网络延迟。表中右列为 2026-11-05 在同机同参数下的复测值。

| 场景 | 吞吐（初测） | 同机复测 |
|---|---|---|
| 单请求（256 tokens） | 53.2 tok/s | 54.3 tok/s |
| 8 并发 × 256 tokens | 411 tok/s | 414.5 tok/s |
| 8 并发 × 512 tokens | 370-405 tok/s（GPU 利用率 99%，有波动） | 402.0 tok/s |

单用户聊天完全流畅；8 并发时 7B 级别接近这张卡的算力上限，再堆并发只会排队。

**长上下文首 token 延迟**（单请求、输出 128 tokens、每个请求带唯一随机前缀以避开 prefix cache，三轮取中位）：

| 输入长度 | TTFT | 总耗时 | 解码速度 |
|---|---|---|---|
| 1024 | 0.57 s | 3.0 s | 18.9 ms/token |
| 2048 | 0.96 s | 3.4 s | 19.3 ms/token |
| 4096 | 1.93 s | 4.4 s | 20.0 ms/token |
| 6144 | 2.97 s | 5.5 s | 20.3 ms/token |
| 7680 | 3.78 s | 6.1 s | 20.4 ms/token |

prefill 成本随输入线性增长，约每 1k tokens 增加 0.47 s；解码速度几乎不随上下文变长而恶化（19-20 ms/token，偶发 32-34 ms 波动来自分块边界与并发）。也就是说 8G 卡上长上下文是"能用，但首字要等"，把整段 8k 文档塞进 prompt 换来的代价是 4 秒左右的首字延迟。

### 4.3 启动参数调优记录

| 参数 | 效果 | 实测 |
|---|---|---|
| gpu-memory-utilization 0.92 | 启动被拒：可用 6.88 / 8.0 GiB，低于所需 7.36 GiB | 失败 |
| gpu-memory-utilization 0.90 | 启动被拒（桌面占用 ~1.1GB，可用仅 6.88GB < 7.2GB） | 失败 |
| gpu-memory-utilization 0.85 | 启动成功，KV 0.98GB | 正常 |
| max-model-len 4096 | KV 1.0GB，并发 4.57x | 正常，但上下文受限 |
| max-model-len 8192 | KV 0.98GB，并发 2.24x | 推荐 |
| enforce-eager 关（走 CUDA Graph） | 图捕获 51 张耗 8 秒、占 0.37 GiB；0.85 / 8192 下 KV = -0.23 GiB，起不来 | 失败 |
| enforce-eager 关 + 4096 上下文 + 0.86 | KV 仍为 -0.13 GiB | 失败 |
| enforce-eager 开 | 启动成功，KV 0.98GB | 8G 卡上唯一可行 |

补充一条 vLLM 0.33 自己在启动日志里算的账：开了 CUDA Graph 内存剖析后，`--gpu-memory-utilization 0.85` 的实际可用当量只有 0.796，要维持同等 KV 得提到 0.904；而这张卡在 Windows 桌面占用下可分配上限只有约 0.86。所以"关掉 enforce-eager 提速"在 8G 消费卡上不是参数没调对，是物理上放不下。

### 4.4 量化位宽对比：AWQ 4-bit vs GPTQ-Int4

同一台机器、同一 prompt、同一启动参数（0.85 / 8192 / enforce-eager），分别服务 AWQ 与 GPTQ 权重（各跑多轮取稳定值）：

| 项 | AWQ 4-bit | GPTQ-Int4 |
|---|---|---|
| 权重体积 | 5.19GB | 5.2GB |
| 显存占用 | 7.19 / 8.0 GiB | 7.19 / 8.0 GiB |
| KV cache | 0.98 GiB（18,368 tokens） | 0.97 GiB（18,240 tokens） |
| 单请求 256 | 53.2 tok/s | 约 48-54 tok/s（有波动） |
| 8 并发 × 256 | 411 tok/s | 406 tok/s |
| 8 并发 × 512 | 370-405 tok/s | 401-412 tok/s |

结论：同参数下两者吞吐落在测量噪声内（±5%），显存占用几乎一致。8GB 卡选哪个看生态即可（AWQ 更新、GPTQ 更老更稳），性能不是分水岭。

> 注：bf16 权重（约 15GB）在 8G 卡上无法加载，未能做同机对比；GGUF 方案需要 llama.cpp 栈，vLLM 0.33 已不提供 GGUF 后端，两者对比另议。

---

## 第 5 章 社区 issue 复现与修复

本章收录三个真实 issue 的复现与修复记录，按"完整修复 → 数值验证 → 校验缺口"排列，分别覆盖三类典型 bug。

### 5.1 案例一：#49497 FlashInfer 采样器 JIT 崩溃（先看现象）

安装 vLLM 0.33.0（纯 wheel，无 CUDA toolkit）后，默认配置启动直接崩溃：

```
INFO  [topk_topp_sampler.py] Using FlashInfer for top-p & top-k sampling.
ERROR [core.py] EngineCore failed to start.
...
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

这是 [vllm-project/vllm#49497](https://github.com/vllm-project/vllm/issues/49497) 描述的问题：FlashInfer 采样器在无法 JIT 编译时没有回退到原生采样器，引擎 warmup 阶段被拖死。WSL2、Docker、只装驱动的 Windows/Linux 等没装 CUDA toolkit 的 wheel 安装用户都会中招。

### 5.2 根因

vLLM 的 `flashinfer_sampler_supported()` 只检查「能否为该 GPU 架构生成代码」（`check_cuda_arch()`），不检查「JIT 需要的 nvcc/ninja 是否真的存在、预编译 cubin 是否版本匹配」。于是 gate 放行，等到 warmup 真正编译时才失败。

本案还叠加一个隐藏坑：flashinfer-cubin 存在但版本（0.6.13）与 flashinfer-python（0.7.2.post1）不匹配，flashinfer 内部会忽略它，但 vLLM 的 `has_flashinfer_cubin()` 只查包是否存在，仍然放行。

### 5.3 本地修复

两个文件，约 60 行：

1. `vllm/utils/flashinfer.py` 新增 `has_matching_flashinfer_cubin()`：cubin 包存在且版本与 flashinfer-python 精确匹配才算可用；
2. `vllm/v1/sample/ops/topk_topp_sampler.py` 新增 `_flashinfer_jit_availability_reason()`，挂到架构检查成功路径上：无匹配 cubin 且 nvcc/ninja 不可用 → 返回原因 → 走既有的 warn-and-fallback（默认）或 raise（显式 `VLLM_USE_FLASHINFER_SAMPLER=1`）。

### 5.4 验证结果

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 默认配置启动 | 崩 | WARNING + 自动回退原生采样，服务正常 |
| 推理请求 | 服务未起来 | HTTP 200，正常输出 |
| 显式 VLLM_USE_FLASHINFER_SAMPLER=1 | 崩 | 清晰报错，提示 unset 环境变量 |

修复后日志：

```
WARNING [topk_topp_sampler.py] FlashInfer top-p/top-k sampling unavailable:
FlashInfer sampling kernels require JIT compilation but no version-matched
flashinfer-cubin is installed and nvcc or ninja is unavailable; falling back.
```

### 5.5 上游状态

该 issue 已有两个 PR 在推进（#49314 只处理「未安装」、#51741 是运行时 try/except fallback），本补丁走 gate 侧精确化，与两者互补。截至 2026-11-05 两个 PR 均为 open、未合并（详见 5.8），所以这个坑还是得自己补。完整复现与修复记录见 `memory/audit/flashinfer-49497-2026-11-05.md`。

### 5.6 案例二：#65401 采样分数一致性（数值 bug 的复现与验证）

第二个案例展示另一类 issue：**候选选择正确、但返回分数错的数值 bug**，以及如何用纯 CPU 复现器在几分钟内确认/验证。

**现象**：`LogitsProcessor.get_top_k_tokens(..., return_log_probs=False)` 对选中的 k 个 logits 先乘 `scale` 再套 `soft_cap`；而 `forward()` 是全词表先 `soft_cap` 再乘 `scale`。两者在 scale≠1 且 soft_cap>0 时不等价。

**实测（vLLM 0.33.0，与本机环境一致，逐位复现）**：

| 量 | 修复前 | 修复后 |
|---|---|---|
| top-k 分数 | [2.971171, 1.748349] | [5.220370, 1.929077] |
| 完整 logits 同 ids | [5.220370, 1.929077] | [5.220370, 1.929077] |
| assert_close | 失败（最大绝对差 2.249） | PASS |

**根因**：`get_top_k_tokens` 尾部两段变换顺序反了；`return_log_probs=True` 分支与 `forward()` 都是 soft_cap → scale，只有普通 top-k 分支是 scale → soft_cap。

**修复**：交换两段顺序（2 行调整，无逻辑变更）。**验证**：issue 自带的纯 CPU reproducer（不下载模型、不占显存）复跑，assert 通过。

**方法论收获**：这类 bug 不需要模型和 GPU，一个 ~30 行的 reproducer 就能锁定，复现成本比性能类 issue 低两个数量级。上游已有人认领修复（本案例用于演示复现与验证方法，不作为独立贡献）。完整记录见 `memory/audit/get-top-k-tokens-65401-2026-11-05.md`。

### 5.7 案例三：#64291 校验缺口——请求该在门口被拦

第三个案例是**校验层缺失**导致的崩溃：请求明明非法，却一路放行到 GPU worker 才炸。

**现象**：当 `--max-logprobs` 是正整数且大于模型 vocab_size 时（默认 20），请求 `logprobs` 数量超过 vocab_size 的请求能通过校验，直到 worker 里 `torch.topk` 才崩（`selected index k out of range`）。`prompt_logprobs` 与 `prompt_logprob_token_ids` 宽度同样有缺口。

**复现（vLLM 0.33.0，纯 mock + 小 CUDA tensor，10 秒内跑完）**：

| 步骤 | 修复前 | 修复后 |
|---|---|---|
| `SamplingParams(logprobs=20)._validate_logprobs()`（vocab=10, max=20） | 通过，不拒绝 | RAISED VLLMValidationError |
| worker 侧 `compute_topk_scores(num_logprobs=20)` | 崩 | 合法请求不再到达 |

**根因**：`_validate_logprobs()` 只对 `max_logprobs == -1` 特判，正值大于 vocab_size 时不 clamp。

**修复**：3 行，`max_logprobs = min(max_logprobs, vocab_size)`。

**方法论收获**：服务端 bug 里"校验层放行、执行层崩溃"是最常见的模式之一，定位方法就是顺着请求路径找"第一个该拒绝却没拒绝的地方"。上游 issue 本身 0 条评论，但一搜发现已经有两个 PR 在修同一个漏洞（#64292、#64361），所以本文的补丁只作为验证材料归档在 `pr-64291/`，不重复提交。完整记录见 `memory/audit/logprobs-64291-2026-11-05.md`。

### 5.8 上游修复追踪（2026-11-05 快照）

三个案例在写作时都还没进上游主干，本地补丁必须留着。用 GitHub REST API 在同一时间点查的状态：

| 编号 | 状态 | 说明 |
|---|---|---|
| #49497 | open，12 条评论 | FlashInfer 采样器启动崩溃的 issue 本身仍未关闭 |
| #49314 | open，未合并 | 候选修复：flashinfer 缺失时回退原生采样 |
| #51741 | open，未合并 | 候选修复：FlashInfer 采样 kernel 编译失败时回退 |
| #64291 | open，0 条评论 | issue 无人讨论，但已有 PR 在修 |
| #64292 | open，未合并 | 校验层拒绝超 vocab 的 logprobs |
| #64361 | open，未合并 | 同上，另一份实现 |

一张表看下来：三个 issue 全是 open、所有候选 PR 也全是 open。至少在 2026-11-05 这个时间点，升级版本并不能自动绕开这些坑，得自己判断要不要打补丁；打补丁前先把 site-packages 原件备份到 `/tmp`，升级后重新确认。

顺带一条方法论：判断"有没有人在修"不能只看 issue 评论区——#64291 评论区是空的，PR 却是有的。要么用 GitHub 搜索接口 `q=repo:<repo>+<issue号>+in:body` 查引用，要么先把 PR 搜一遍再动手，否则很容易做一遍重复劳动。

---

## 第 6 章 8GB 显存优化清单（SOP）

拿到一张 8GB 卡，按此顺序做：

1. **选 4-bit 量化权重**（AWQ/GPTQ 均可，7B 约 5.2GB），别碰 bf16；
2. **gpu-memory-utilization 从 0.85 起调**，Windows 下 0.9 可能被拒；
3. **max-model-len 设 8192**，这是余量上限；需要更长上下文用 `--max-num-seqs` 降并发换 KV；
4. **enforce-eager 必开**：8G 卡关掉 CUDA Graph 后 KV cache 余量为负、服务起不来（4.3 节有实测数据）；
5. **别超 8 并发**，再多只是排队；
6. **采样器走原生**（`VLLM_USE_FLASHINFER_SAMPLER=0`）——或者用第 5 章的修复让引擎自己回退；
7. **长回答用 HTTP 流式输出**，降低首 token 感知延迟。

---

## 第 7 章 局限与可复现性

- **单卡双方案**：本文数据仅覆盖 RTX 4060 Laptop 8GB 与 Qwen2.5-7B（AWQ/GPTQ），换卡/换模型需重测；
- **Windows 特有**：0.9 利用率被拒是 Windows 桌面显存占用导致的，纯 Linux 服务器上余量不同；
- **对比范围**：enforce-eager 开关已补做同机对比（结论是 8G 卡上不能关闭）；bf16 因显存不足、GGUF 因 vLLM 无后端，二者无法同机对比；
- **性能波动**：并发场景吞吐在 400 tok/s 上下有 ±5% 波动，对比以多轮稳定值为准；
- **时间敏感**：vLLM 版本迭代快，第 5 章的 issue 状态以本文写作时（2026-11-05）为准，跟进前请先核对上游最新状态。

---

## 附录 A 复现脚本

```bash
# 启动
vllm serve <model> --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --port 8000 --host 0.0.0.0 --enforce-eager

# 单请求
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

## 附录 B 完成情况

- [x] 更多 8G 相关 issue 复现（已 3 个：#49497、#65401、#64291）
- [x] 量化方案对比（AWQ vs GPTQ 已完成；bf16 显存不足，GGUF 无 vLLM 后端）
- [x] 长上下文（4k-8k）延迟曲线（4.2 节，2026-11-05 补）
- [x] enforce-eager 开关对比实测（4.3 节：8G 卡上关闭即启动失败）
- [x] 上游 PR 结果追踪（5.8 节，2026-11-05 快照）
- [x] #64291 修复整理成验证材料（`pr-64291/`：补丁 + 复现脚本；提交前发现上游已有 #64292、#64361 两个 PR，未重复提交）

唯一没做的一项：多卡或更大显存卡的同参数复测——没有硬件，等有了再补。

## 附录 C 上下文上限的显存账（2026-11-05 复测）

同一台机器、桌面与浏览器占用约 1.1 GB 时（启动时空闲显存 6.88 / 8.0 GiB），固定 `--gpu-memory-utilization 0.85`，逐档试 `max-model-len`：

| max-model-len | 可用 KV 池 | KV 容量 | 单请求并发 | 结果 |
|---|---|---|---|---|
| 4096 | 0.88 GiB | 16,560 tokens | 4.04× | 正常 |
| 5280 | — | — | — | 正常（实测可跑） |
| 8192 | 0.28–0.36 GiB | — | — | 起不来：需要 0.44 GiB，剩余不够 |
| 16384 | 同上 | — | — | 起不来 |

8K 启动失败时 vLLM 自己给的估计：按当时可用内存，单请求最大长度约 5280 tokens。实测 5280 上下文、4,252 token 提示词，预填充 + 解码 64 token 共 3.47 s（约 18.5 tok/s）。

两条结论：

- **8192 不是硬上限，是桌面让出来的上限**。第 4.3 节记录的 KV 0.98 GiB（18,368 tokens）对应桌面占用更低的时刻；桌面吃掉约 1.1 GB 时，8G 卡的实际天花板是 5280。对外说"8G 能跑 8K"必须带上桌面占用这个前提，否则换台机器复现不出来。
- **fp8 KV 能压一半，但要 nvcc**。`kv_cache_dtype=fp8` 下引擎初始化通过（12,592 tokens 池、8192 请求 1.54× 并发），但生成阶段走 JIT 编译，直接报 `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`；环境里装好 CUDA toolkit 才可用。没装之前，8K 以上只能靠更激进压缩或更小的模型，本文未测。
