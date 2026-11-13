# Biomni Research Agent

A tool-calling research agent for six biomedical tasks. It answers each question by querying biomedical databases
and online APIs, and is trained with GRPO on
[SageMaker Training Sessions](../../src/agentcore_rl_toolkit/backends/sagemaker/SETUP.md).

| Task (`task_name`) | Question | Answer |
| --- | --- | --- |
| `lab_bench_dbqa` | Multiple choice over gene sets, interactions, disease evidence, ClinVar and protein records | Option letter |
| `lab_bench_seqqa` | Multiple choice on DNA and protein sequences: ORFs, primers, digests, cloning | Option letter |
| `screen_gene_retrieval` | The candidate gene that is a hit in a described CRISPR screen | Gene symbol |
| `gwas_causal_gene_gwas_catalog` | The causal gene at a GWAS locus, from the GWAS Catalog and Open Targets | Gene symbol |
| `gwas_causal_gene_opentargets` | The causal gene at a GWAS locus, from Open Targets | Gene symbol |
| `gwas_variant_prioritization` | The most likely causal variant among candidates, from Open Targets | rsID |

The agent runs on AgentCore Runtime. Its tools run in a separate tool service, one process per task family
(`dbqa`, `seqqa`, `screen`, `gwas`), on a CPU instance that holds the environment data and your NCBI API keys:

```
trainer / caller ──> AgentCore Runtime (rl_app.py): episode loop ──> model (base_url in _config)
                                     └──> tool service (tool_service.py) ──> environment data, NCBI, UniProt, Ensembl,
                                                                              Open Targets, ...
```

The reward is 1 when the final answer, in its canonical form, matches the label, and 0 otherwise.

## Results

Reference run results (same agent, data and hyperparameters, trained on SageMaker MTRL). Qwen3.6-27B, thinking off,
pass@1 over 8 samples per question:

| Model | dbqa | seqqa | screen | GWAS Catalog | OT causal gene | OT variant | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B | 0.715 | 0.625 | 0.720 | 0.607 | 0.553 | 0.259 | 0.580 |
| + LoRA r32, step 42 | 0.902 | 0.870 | 0.915 | 0.655 | 0.758 | 0.637 | 0.790 |

## Install

Use Python 3.12. From this directory:

```bash
uv sync
```

## Prepare the environment data

```bash
uv run python prepare_data.py --out data
```

The script downloads about 2.2 GB from the public sources (GWAS Catalog 2026-03-17, BioGRID ORCS 2.0.18, the Biomni
data lake, MSigDB 2023.2, MONDO, HPO, ClinVar, RefSeq, MANE 1.5, Ensembl 116 and two GenBank records) into
`data/raw/` and builds about 2.3 GB of environment data under `data/`. Delete `data/raw/` afterwards. Rerun a single
step with `--only`, e.g. `--only clinvar`.

## Start the tool services

On the CPU instance, with the environment data in `data/` (or set `BIOMNI_DATA_DIR`) and your
[NCBI API keys](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317), comma-separated:

```bash
export NCBI_API_KEYS=KEY1,KEY2
uv run python tool_service.py dbqa   --port 8101 &
uv run python tool_service.py seqqa  --port 8102 &
uv run python tool_service.py screen --port 8103 &
uv run python tool_service.py gwas   --port 8104 &
```

Each service binds to `127.0.0.1` by default. For the Runtime, start them with `--host <the instance's private IP>`
and allow TCP 8101–8104 only from the Runtime's security group.

## Run the agent

Point the agent at the tool services and start it locally:

```bash
export TOOL_SERVICE_DBQA=http://127.0.0.1:8101 TOOL_SERVICE_SEQQA=http://127.0.0.1:8102 \
       TOOL_SERVICE_SCREEN=http://127.0.0.1:8103 TOOL_SERVICE_GWAS=http://127.0.0.1:8104
uv run python rl_app.py
```

From another terminal, send a task. `_config` names an OpenAI-compatible server for the model, here Qwen3.6-27B
served by vLLM with tool calling enabled; to run a trained checkpoint, serve the base model with its LoRA adapter
and set `model_id` to the adapter's name:

```bash
curl -X POST http://127.0.0.1:8080/invocations -H "Content-Type: application/json" -d '{
  "task_name": "gwas_variant_prioritization",
  "prompt": "<question>",
  "answer": "<label, optional>",
  "_config": {"base_url": "http://127.0.0.1:8000/v1", "model_id": "Qwen/Qwen3.6-27B"}
}'
```

The response holds `reward`, the parsed `answer` and the final `response`. The agent sets its own sampling
(temperature 0.7, top-p 0.8, top-k 20, thinking off) and 8,192 output tokens per turn in a 64K context.

## Deploy to AgentCore Runtime

The Runtime must be in VPC mode, in a network that reaches the tool services and the training host:

```bash
agentcore configure --entrypoint rl_app.py --name biomni_research_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent biomni_research_agent \
  --env TOOL_SERVICE_DBQA=http://TOOL_HOST:8101 --env TOOL_SERVICE_SEQQA=http://TOOL_HOST:8102 \
  --env TOOL_SERVICE_SCREEN=http://TOOL_HOST:8103 --env TOOL_SERVICE_GWAS=http://TOOL_HOST:8104
```

## Train

Training rows are payloads with `task_name`, `prompt` and `answer`, in a Parquet file with a `payload` column:

```python
import pandas as pd

pd.DataFrame({"payload": [{"task_name": "lab_bench_seqqa", "prompt": "...", "answer": "C"}]}).to_parquet("train.parquet")
```

Follow the [SageMaker backend setup](../../src/agentcore_rl_toolkit/backends/sagemaker/SETUP.md) for the training
environment and AWS resources. On an EC2 instance in the Runtime's VPC, whose security group allows the gateway port
only from the Runtime's security group:

```bash
cp config.example.yaml config.yaml   # fill in the AWS resources, agent_runtime_arn and dataset_path
python -m agentcore_rl_toolkit.backends.sagemaker.train_grpo --config config.yaml
```

`config.example.yaml` holds the reference run's settings: LoRA rank 32 / alpha 64, batch 64 questions × 8 rollouts,
learning rate 5e-5, PPO loss, one epoch, thinking off, 64K context.
