---
title: Async RL Performance Optimization
description: Improve training throughput with asynchronous reinforcement learning
---

_By [Danylo Vashchilenko](https://github.com/hellodanylo) · October 13, 2026_

:::note[TL;DR]
* 🚀 While training [SWE Gym](../../examples/openhands-swegym-agent) agent, we show that asynchronous RL enables higher learning throughput, leading to ~22% higher reward improvement per GPU-hour of training.
* ℹ️ We show that the efficiency increase is explained by higher LLM inference concurrency and throughput.
* ⏭️ We observe that KV cache remains underutilized (time-average at ~25%), which is an opportunity for further optimization.
:::

## RL Throughput and Concurrency of Inference and Training

All other things being equal, we prefer an RL system with higher throughput, as measured in trajectories per GPU-hour. Throughput and concurrency are positively correlated up to a certain level of concurrency (the saturation point).

For the inference servers, KV cache capacity is the bottleneck in SWE and other long-context tasks. 
For example, for Qwen3 Coder 30B on p6 node, there is ~5M tokens of capacity, which is only 78 trajectories at 64K context. 
The actual concurrency typically peaks right after a rollout batch is dispatched and reaches the low right before the next rollout batch is dispatched.
Selecting a lower timeout threshold (which reduces the tail) can lead to significant throughput gains. 
In order to avoid thrashing the prefix cache, the concurrency is set in terms of active trajectories, not in terms of active inference requests.

In this report, we will specifically compare (1) on-policy, 1 mini-batch per step, and (2) off-policy, 2 mini-batches per step. We expect that **increasing mini-batch count per step will increase rollout throughput**, because higher inference concurrency will bring the hardware closer to the utilization saturation point.

![Sync RL](../../../assets/blog/async-rl-performance-optimization/sync_rl.png)

For the training servers, GPU memory (activations, gradients, optimizer state) is also the bottleneck, but the token batch size is typically significantly higher than the saturation point, so the latency of the update step is roughly linear in the count of tokens. For long trajectories, the training server typically needs [memory saving methods](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) like Model/Context Parallelism, Activation Recomputation and Offloading, which reduce throughput. Quantization can significantly improve throughput if it allows to turn off the memory saving methods without hitting OOMs.

It's previously been widely discussed that **RL is rollout-bound, but that's not always true for multi-turn RL** like SWE. The inference server encodes the new context tokens (e.g. prompt and tool responses) in parallel and decodes the agent response tokens sequentially. The training server encodes all tokens in parallel, but also does the backward pass for each token. Therefore, the rollout throughput is positively correlated with the non-agent fraction of tokens in the context, while the trainer throughput does not depend on this ratio.

| Server | Agent Tokens | Environment Tokens |
| --- | --- | --- |
| Training | parallel fwd+bwd | parallel fwd+bwd |
| Inference | sequential fwd (slower) | parallel fwd (faster) |

What we observe in actual SWE training runs is that in an average trajectory only ~20% of tokens are from the agent. In this regime, the **inference and training servers can have roughly the same throughput per GPU**. When the inference concurrency is actually maximized past saturation, the inference throughput can even be higher than trainer's (e.g. trainer-bound regime). This observation is highly important for prioritizing training efficiency research, and for allocating hardware resources between training and inference servers in experiments. 

Dynamic sampling (e.g. [DAPO](https://arxiv.org/abs/2503.14476)) reduce the useful throughput of the inference servers proportionally with the frequency of zero-advantage groups. In short, the botteleneck is an async RL pipeline depends on many charectiristics of the workload, but it's certainly not always rollout-bound.

Using the framework of this section, we expect the disaggregated cluster to have higher throughput than a colocated cluster, because reducing inference servers count increases the effective inference concurrency. And we expect the async RL pipeline to have higher throughput than sync RL pipeline, because rollout batches can overlap, leading to higher inference concurrency. 

![Sync RL](../../../assets/blog/async-rl-performance-optimization/async_rl.png)

With the knowledge that the inference and trainer throughput are roughly the same, we choose to have 1:1 ratio of GPUs allocated to trainer and rollout in the disaggregated cluster. We will specifically compare (1) a colocated cluster with 2 hybrid model replicas and (2) a disaggregated cluster with 1 training replica and 1 inference replica. 

## Training Results

First, let's review the trainer and rollout throughput measurements of these configurations. As expected, we observe a significant throughput speedup (up to ~1.42x) from using off-policy and async RL, which is explained by the increase of average inference request concurrency. In the sync-1 baseline, the average concurrency of ~11 does not saturate the GPU throughput. Additionally, we observe that average KV cache utilization in all cases is <25%, which means that there is 4x headroom for higher concurrency, which is an opportunity for further optimization.

| **Training Cluster Config** | **Throughput (kilo-tokens/minute/GPU)** | **Throughput Speedup** | **Inference Concurrency per Replica** |
| --- | --- | --- | --- |
| Sync 1 step | 27.82 | 1 | 11.03 |
| Sync 2 steps | 34.47 | 1.24x | 15.20 |
| Async 1 step | 38.49 | 1.38x | 33.97 |
| Async 2 steps | 39.39 | 1.42x | 38.22 |

We know from the past RL research that off-policy learning can be less efficient per optimizer step, so next we measure whether reward
has the same rate of change per 100 optimizer steps. We observe that Async-1 and Sync-2 configurations reduce the rate of reward improvement, while Async-2 configuration increases it.

| **Training Cluster Config** | **Reward Improvement Rate (per 100 optimizer steps)** | **Reward Improvement Speedup** |
| --- | --- | --- |
| Sync 1 step | 5.98pp | 1 |
| Sync 2 steps | 5.71pp | 0.96x | 
| Async 1 step | 4.55pp | 0.76x |
| Async 2 steps | 7.79pp | 1.30x |

Finally, we would like to combine the two parts of this analysis -- throughput per GPU and reward improvate rate -- giving us a measurement of the reward change per GPU-hour. In the following table, we calculate the reward's rate of change per 640 GPU-hours. We observe that Async-2 configuration learns 1.22x faster and cheaper than the Sync-1 baseline.

| **Training Cluster Config** | **Reward Improvement Rate (per 640 GPU-hours)** | **Reward Improvement Speedup** |
| --- | --- | --- |
| Sync 1 step | 7.17pp | 1 |
| Sync 2 steps | 7.86pp | 1.10x | 
| Async 1 step | 7.05pp | 0.98x |
| Async 2 steps | 8.73pp | 1.22x |


We plot the learning curve of the Sync-1 and Async-2 training runs against the GPU-hours used to observe that Async-2 configuration has been ahead of the Sync-1 run during most of the training runtime.

![Reward Curve](../../../assets/blog/async-rl-performance-optimization/reward_curve.png)