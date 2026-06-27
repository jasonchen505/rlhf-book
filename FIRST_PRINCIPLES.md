# RLHF Book 第一性原理解读: 从 LM 基础到后训练

> 按照 Language Models Overview -> LM Head -> Softmax -> Training Example -> Probabilities -> Masks -> Decoding -> Cross-Entropy -> Optimization -> Pipeline -> KL/Entropy -> Sigmoid -> RL Framing -> Post-Training Tools 的脉络, 从最底层的数学和代码出发, 重新解读整个 rlhf-book 项目。

---

## 1. Language Models Overview

### 1.1 语言模型的本质

语言模型的核心任务: 给定前面的 token, 预测下一个 token 的概率分布。

自回归分解 (书中 Appendix A @eq:llming):

```
P_theta(x) = prod_{t=1}^{T} P_theta(x_t | x_1, ..., x_{t-1})
```

关键洞察: **整个 LLM 就是在学习这个条件概率分布**。预训练和后训练都基于同一个 loss -- next-token prediction。

### 1.2 预训练 vs 后训练的连续性

| 维度 | 预训练 | 后训练 (SFT) |
|------|--------|-------------|
| 数据 | 互联网文本 | 指令-回答对 |
| 目标 | 学习语言统计规律 | 学习回答格式和风格 |
| Mask | 所有 token 都算 loss | 只有 response tokens 算 loss |
| Batch Size | 很大 (1024+) | 较小 (256) |
| LR | 较高 (3e-4) | 较低 (1e-5) |

### 1.3 为什么自回归分解如此强大

自回归分解的威力在于: **任何文本都可以用同一个框架处理**。对话、论文、代码都是 token 序列。这为后训练 (SFT -> RM -> RL) 提供了统一的基础 -- 所有阶段都在操作同一个概率模型。

---

## 2. The LM Head

### 2.1 定义

书中 Appendix A:

> The LM head is a final linear projection layer that maps from the model internal embedding space to the tokenizer space (a.k.a. vocabulary).

```
Input tokens -> Embedding -> N x Transformer Block -> hidden_states -> LM Head -> logits
```

LM Head = nn.Linear(hidden_size, vocab_size)

### 2.2 代码中的三种 LM Head 用法

**标准 LM Head (SFT/预训练)**: 输出 vocab_size 维 logits

```python
# instruction_tuning/utils.py:123
out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
# out.logits shape: (batch, seq_len, vocab_size)
```

**Reward Model Head**: 替换为 1 维标量输出

```python
# policy_gradients/utils.py:117-119 (PPO value model)
val_model.lm_head = nn.Linear(val_model.lm_head.in_features, 1, bias=False)

# reward_models/train_preference_rm.py:148
class PreferenceRewardModel(BaseRewardModel):
    def __init__(self, model_id):
        super().__init__(model_id, head_dim=1)  # output 1-dim
```

**关键洞察**: 后训练中, 同一个 backbone + 不同 head = 不同用途。RM head 从 EOS token 的 hidden state 输出标量 reward; PPO value head 从每个 token 输出 state value。

---

## 3. Softmax & Log-Probabilities

### 3.1 Softmax

```
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

性质: 输出在 (0,1), 和为 1, 保持相对大小。

### 3.2 为什么用 log-softmax

1. exp(z_i) 可能溢出
2. 多个小概率相乘会下溢到 0

```
log_softmax(z_i) = z_i - log(sum_j exp(z_j))
```

代码中统一使用 F.log_softmax(logits, dim=-1)。

### 3.3 gather 操作

```python
# policy_gradients/utils.py:258-260
log_probs = F.log_softmax(logits, dim=-1)            # (batch, seq, vocab)
targets = sequence_ids[:, 1:].unsqueeze(-1)           # (batch, seq, 1)
target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
# result: (batch, seq)
```

**这就是模型对实际 token 赋予概率的数学表达。整个后训练都在操作这个量。**

---

## 4. Anatomy of an LM Training Example

### 4.1 SFT 样本的三个 tensor

- **input_ids**: 完整 token 序列 (system + user + assistant)
- **attention_mask**: 1=真实 token, 0=padding
- **labels**: 和 input_ids 相同, 但 prompt 部分设为 -100 (IGNORE_INDEX)

```python
IGNORE_INDEX = -100

def compute_loss(model, batch):
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
```

### 4.2 shift 的含义

- 位置 t 的 logits 是模型基于 token 0..t 预测的下一个 token 分布
- 位置 t 的 label 是位置 t+1 的实际 token
- 所以 logits[:, :-1] 和 labels[:, 1:] 对齐

### 4.3 DPO 样本

DPO 样本包含两个序列 (chosen 和 rejected), 各自有 response_mask: 1=response token, 0=prompt token。

---

## 5. Computing LLM Probabilities

### 5.1 序列 log 概率

```
log P(x) = sum_{t=1}^{T} log P(x_t | x_<t)
         = sum_{t=1}^{T} log_softmax(logits_t)[x_t]
```

这就是代码中 per_token_logps.sum(dim=-1) 的数学含义。

### 5.2 两种聚合方式

- **Sum (DPO 默认)**: 真正的 log 概率, 但有长度偏差
- **Average (SimPO/DPO-Norm)**: 消除长度偏差, 但不再是真正 log 概率

```python
if average_log_prob:
    return per_token_logps.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
else:
    return per_token_logps.sum(dim=-1)
```

### 5.3 log-prob 在后训练中的角色

| 阶段 | 用途 |
|------|------|
| SFT | cross-entropy = -log P(correct token) |
| DPO | log-ratio = log P_policy - log P_ref |
| PPO/GRPO | ratio = exp(log_pi_new - log_pi_old) |
| KL | sum of (log_pi - log_pi_ref) per token |

**log-prob 是连接所有后训练算法的核心数学对象。**

---

## 6. Three Common Masks in Post-Training

### 6.1 attention_mask

告诉 Transformer 哪些位置有真实 token (1), 哪些是 padding (0)。所有模块都使用。

### 6.2 labels / IGNORE_INDEX

SFT 中, prompt tokens 的 label 设为 -100, PyTorch CrossEntropyLoss 自动忽略。

### 6.3 response_mask / action_mask

DPO 和 RL 中, 标识哪些 token 是 response (1), 哪些是 prompt (0)。

### 6.4 三种 mask 对比

| Mask | 用途 | 被 mask 的位置 | 场景 |
|------|------|--------------|------|
| attention_mask | Attention 机制 | padding -> -inf | 所有 |
| labels | Loss 计算 | prompt -> 忽略 | SFT |
| response_mask | Log-prob 聚合 | prompt -> 0 | DPO, RL |

---

## 7. A Small Decoding Review

### 7.1 生成即采样

LLM 生成: softmax -> 采样 -> 追加 -> 重复, 直到 EOS 或 max_length。

### 7.2 采样参数

| 参数 | 作用 | 默认值 |
|------|------|-------|
| temperature | 缩放 logits: z/T | 0.6 (GRPO) |
| top_p | nucleus sampling | 0.95 |
| top_k | top-k sampling | 20 |
| min_p | 过滤低概率 token | 0.0 |

### 7.3 RL 训练需要 temperature > 0

temperature=0 时同一 prompt 永远生成同一回答, 无法探索。GRPO 的 num_rollouts 依赖采样多样性。

---

## 8. Training an LM: Cross-Entropy

### 8.1 Cross-Entropy = NLL

```
H(p, q) = -log P_theta(x_t | x_<t)  (when p is one-hot)
```

### 8.2 各模块的 loss

```python
# SFT: standard CE
loss = F.cross_entropy(logits, labels, ignore_index=-100)

# RM: log-sigmoid pairwise
loss = -F.logsigmoid(r_chosen - r_rejected).mean()

# DPO: log-sigmoid on log-ratios
loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

# REINFORCE: advantage-weighted log-prob
loss = -(log_probs * advantages).mean()
```

**所有后训练 loss 都是某种形式的交叉熵: 让好的更可能, 坏的更不可能。**

---

## 9. Optimization & Fine-Tuning

### 9.1 AdamW

所有训练用 AdamW, 搭配 linear warmup + linear decay LR schedule。

### 9.2 Gradient Clipping

所有模块用 max_norm=1.0: RL 训练中 reward 信号 noisy, 梯度爆炸风险高。

### 9.3 Gradient Checkpointing

所有模块默认开启, 减少 ~30-40% activation 内存, 用时间换空间。

---

## 10. Pretraining to Midtraining to SFT Pipeline

### 10.1 三阶段流水线

| 阶段 | 数据 | 方法 | 效果 |
|------|------|------|------|
| SFT | ~1M 指令对 | cross-entropy + prompt masking | 学会回答格式 |
| PreFT | ~1M 偏好对 | RLHF 或 DPO | 对齐人类偏好 |
| RLVR | ~10K 可验证题 | RL + verifiable rewards | 提升推理能力 |

### 10.2 经典案例

- **InstructGPT**: SFT(10K) -> RM(100K) -> RLHF(100K)
- **Tülu 3**: SFT(1M) -> On-policy PrefData(1M) -> RLVR(10K)
- **DeepSeek R1**: Cold-start(100K) -> RLVR(large) -> RS+SFT(800K) -> Mixed RL

---

## 11. Probability Essentials: KL Divergence & Entropy

### 11.1 KL 散度

```
D_KL(P || Q) = E_{x~P} [log P(x) - log Q(x)]
```

性质: D_KL >= 0, D_KL=0 iff P=Q, 不对称。

### 11.2 RLHF 中的 KL 方向

- **Reverse KL** (D_KL(pi_RL || pi_ref)): 标准 RLHF 选择, 当新模型在 ref 低概率处放高概率时惩罚大。
- **Forward KL** (D_KL(pi_ref || pi_RL)): 更接近 distillation。

### 11.3 三种 KL 估计器

```python
# kl1: -log_ratio
# kl2: 0.5 * (log_ratio)^2
# kl3: (exp(log_ratio) - 1) - log_ratio  [default, most stable]
```

### 11.4 Entropy

H(P) = -sum P(x) * log P(x)。衡量模型不确定性。RL 中 entropy bonus 鼓励探索。

---

## 12. Sigmoid & Pairwise Likelihood

### 12.1 Sigmoid

```
sigmoid(x) = 1 / (1 + exp(-x))
log sigmoid(x) = -log(1 + exp(-x)) = -softplus(-x)
```

### 12.2 Bradley-Terry 模型

```
P(y_c > y_r | x) = sigmoid(r_chosen - r_rejected)
```

### 12.3 DPO 的隐式奖励

```
r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))
P(y_c > y_r) = sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

```python
chosen_logratios = policy_chosen_logps - ref_chosen_logps
rejected_logratios = policy_rejected_logps - ref_rejected_logps
logits = chosen_logratios - rejected_logratios
losses = -F.logsigmoid(self.beta * logits)
```

---

## 13. Reinforcement Learning Framing (MDP)

### 13.1 标准 RL vs RLHF

| 维度 | 标准 RL | RLHF |
|------|--------|------|
| Policy | 从零学习 | 从预训练 fine-tune |
| Reward | 环境函数 | 学习的 RM |
| Transition | 有 | 无 (bandit) |
| Action | 单步 | 整个 completion |
| Discount | gamma < 1 | gamma = 1 |

### 13.2 策略梯度定理

```
Delta_theta proportional to Psi_t * grad_theta log pi_theta(a_t | s_t)
```

- grad log pi: 指向让 a_t 更可能的方向
- Psi_t: advantage (这个动作有多好)

```python
# REINFORCE loss
loss = -(log_probs * advantages)
```

---

## 14. Transitioning Tools into Post-Training

### 14.1 基础工具到后训练的映射

| 基础工具 | 后训练用途 |
|---------|-----------|
| LM Head | SFT/RM/Value head 的基础 |
| Softmax | logits -> 概率分布 |
| Log-probs | 策略梯度/DPO/KL 的核心 |
| Cross-Entropy | SFT/RM loss |
| Prompt Masking | 只学 response |
| response_mask | DPO/RL 的 response 计算 |
| Decoding | RL rollout generation |
| KL Divergence | 防 over-optimization |
| Sigmoid | BT RM/DPO loss |
| RL/MDP | 策略梯度框架 |

### 14.2 核心洞察

**所有后训练算法都在做同一件事: 调整模型的概率分布, 让好的 token 更可能, 坏的 token 更不可能。**

区别在于:
1. 好和坏的定义 (人工标注 vs RM 打分 vs 可验证奖励)
2. 调整幅度控制 (KL/clipping/beta)
3. 数据来源 (离线 vs 在线)

这就是为什么整个项目可以用同一套代码框架实现所有算法 -- 它们共享相同的数学基础。

---

*本文档从第一性原理出发, 重新解读 RLHF Book 的所有核心概念。*
*配套: INTERVIEW_PREP.md, INTERVIEW_DEEP_DIVE.md, REPRODUCTION_PLAN.md, REPRODUCTION_LEARNING.md*