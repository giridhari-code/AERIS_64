# Company platform status — what “full” means here

## Delivered in this package

| Capability | Status |
|------------|--------|
| NeuroField architecture (paper-faithful core) | Yes |
| Safetensors checkpoints | Yes |
| BPE + char tokenizers | Yes |
| Train / eval scripts | Yes |
| FastAPI + Swagger | Yes |
| Chat UI | Yes |
| Streaming SSE | Yes |
| API keys + rate limit | Yes |
| /v1/models | Yes |
| Metrics / health | Yes |
| Docker / K8s stubs | Yes |
| India / Hindlish data path | Yes |
| Model card + API docs | Yes |

## Explicitly NOT included (cannot fake)

| Capability | Why missing |
|------------|-------------|
| 7B–70B+ pretrained weights | Needs huge compute & data |
| Trillion-token web crawl training | Not available in this environment |
| Human-level multilingual fluency | Needs large multilingual pretraining |
| RLHF / constitutional fine-tuning at scale | Separate large program |
| Multi-region HA SaaS | Infra outside this repo |

**Bottom line:** This is a **company-shaped product shell** around a **real research architecture**, suitable for demos, internal APIs, and continued training — not a competitor to frontier LLMs out of the box.

## Recommended next investments

1. More domain data + longer `train_company.py` runs  
2. BPE vocab 2k–8k + `d_model` 256–512 on GPU  
3. Redis session store  
4. Full red-team + content policy layer  
5. Continuous eval in CI  
