from unittest.mock import Mock
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

model_config = Mock()
model_config.max_logprobs = 20
model_config.get_vocab_size.return_value = 10
try:
    SamplingParams(logprobs=20)._validate_logprobs(model_config)
    print("validate: PASSED (bug: no rejection)")
except Exception as e:
    print("validate: RAISED", type(e).__name__, str(e)[:120])

logits = torch.randn(2, 10, device="cuda")
sampled = torch.tensor([3, 5], device="cuda")
try:
    compute_topk_scores(logits, num_logprobs=20, sampled_token_ids=sampled)
    print("topk: ok")
except Exception as e:
    print("topk: CRASH", type(e).__name__, str(e)[:120])