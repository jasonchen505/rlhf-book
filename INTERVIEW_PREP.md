# LLM 后训练 & RLHF 面试深度准备手册

> 基于 Nathan Lambert 的 [RLHF Book](https://rlhfbook.com) 项目系统梳理，面向 LLM 算法实习面试。
> 本文档覆盖项目所有章节核心知识 + 代码实现细节 + 可被面试官深挖的考察点。

---

## 目录

1. [项目全景概览](#1-项目全景概览)
2. [RLHF 三阶段流水线：SFT -> RM -> RL](#2-rlhf-三阶段流水线)
3. [指令微调 (SFT) 深度要点](#3-指令微调-sft-深度要点)
4. [奖励模型 (Reward Modeling) 深度要点](#4-奖励模型-reward-modeling-深度要点)
5. [策略梯度 (Policy Gradients) 深度要点](#5-策略梯度-policy-gradients-深度要点)
6. [直接对齐算法 (DPO 及变体) 深度要点](#6-直接对齐算法-dpo-及变体-深度要点)
7. [拒绝采样 (Rejection Sampling) 深度要点](#7-拒绝采样-rejection-sampling-深度要点)
8. [推理模型与 RLVR](#8-推理模型与-rlvr)
9. [工具调用与 Agent](#9-工具调用与-agent)
10. [过优化与正则化](#10-过优化与正则化)
11. [偏好数据与合成数据](#11-偏好数据与合成数据)
12. [评估体系](#12-评估体系)
13. [Character Training 与产品化](#13-character-training-与产品化)
14. [代码实现细节速查](#14-代码实现细节速查)
15. [面试深挖问题集](#15-面试深挖问题集)
16. [自我介绍框架建议](#16-自我介绍框架建议)

---

## 1. 项目全景概览

### 1.1 项目结构

```
rlhf-book/
+-- book/chapters/          # 17章 + 3个附录的 Markdown 源文件
+-- code/                   # 参考实现
|   +-- instruction_tuning/ # SFT 训练
|   +-- policy_gradients/   # PPO/REINFORCE/GRPO/RLOO 等 10 种算法
|   +-- reward_models/      # 偏好RM、ORM、PRM 训练
|   +-- direct_alignment/   # DPO/IPO/SimPO/KTO 等 8 种算法
|   +-- rejection_sampling/ # Best-of-N 拒绝采样
+-- teach/course/           # 课程讲义 (8 节课)
+-- diagrams/               # 图表源文件
```

### 1.2 核心思想链

```
Base Model -> SFT(指令微调) -> RM(奖励模型) -> RL(PPO/GRPO等) -> 最终模型
                   |                |               |
             学会回答格式      学会人类偏好     用偏好信号优化策略
```

**现代后训练(2025+)的完整栈：**
1. **IFT/SFT**：教格式，建立指令遵循能力的基础 -> 学习语言的 *特征*
2. **PreFT(偏好微调)**：通过 RLHF 等方法对齐人类偏好 -> 学习语言的 *风格*
3. **RLVR**：用可验证奖励做 RL 提升推理/代码/数学 -> 提升 *能力*

---

## 2. RLHF 三阶段流水线

### 2.1 阶段一：SFT（指令微调）

**核心原理**：用 next-token prediction loss 在指令-回复数据上继续训练，让 base model 学会以 QA 格式回答问题。

**关键实现细节（高频考点）**：
- **Chat Template**：用特殊 token（如 `<|im_start|>`, `<|im_end|>`）包裹 system/user/assistant 角色信息，通过 Jinja 模板（`apply_chat_template`）将消息列表转为 token 序列
- **Prompt Masking**：只对 assistant 回复部分计算 loss，prompt 部分被 mask 掉（模型不学预测用户问题）
- **多轮对话 Masking**：两种策略 -- (1) Final-turn only：只在最后一个 assistant turn 算 loss (2) Mask user turns only：所有 assistant turn 都算 loss
- **学习率**：SFT 通常比预训练低 1-2 个数量级（如预训练 3e-4 -> SFT 1e-5），因为数据量更小、分布更窄
- **Batch Size**：比预训练小得多（如 OLMo 2 预训练用 1024/2048，SFT 只用 256 prompts）
- **训练时长**：通常 1-3 个 epoch，避免过拟合

**面试深挖点**：
- Q: 为什么 SFT 用和预训练相同的 loss 但效果不同？
  A: 核心在于数据分布不同（指令-回答格式 vs 原始文本）+ prompt masking + 更小的 batch size + 更低的学习率
- Q: Chat Template 为什么重要？格式错误会怎样？
  A: 模型通过特定 token 序列识别角色切换。格式不一致会导致模型无法正确区分 system/user/assistant，推理时行为混乱
- Q: 为什么 SFT 数据不在于多而在于精？
  A: 论文 LIMA 表明 1000 条高质量数据就能达到不错的 chat 效果。SFT 本质上是让模型学会格式，能力主要来自预训练

### 2.2 阶段二：奖励模型训练

**核心原理**：在偏好数据（chosen vs rejected 成对对比）上训练一个模型输出标量奖励分数。

**Bradley-Terry 模型（核心公式）**：
```
P(y_c > y_r | x) = sigmoid(r_theta(y_c|x) - r_theta(y_r|x))
Loss = -log sigmoid(r_chosen - r_rejected)
```

等价写法（softplus 形式）：
```
Loss = log(1 + exp(r_rejected - r_chosen))
```

**架构**：在 LLM 的最后一个 hidden state 上加一个线性层，输出标量 reward。

**关键实现细节**：
- 用 EOS token（最后一个非 padding token）的 hidden state 作为序列表示
- 通过线性头 `nn.Linear(hidden_size, 1)` 得到标量分数
- 通常只训练 1 个 epoch 防止过拟合
- 代码核心：`loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()`

**奖励模型变体（高频考点）**：

| 类型 | 描述 | 应用场景 |
|------|------|---------|
| **Bradley-Terry RM** | 标量偏好模型，对 (chosen, rejected) 学习差值 | 标准 RLHF 流程 |
| **ORM (Outcome RM)** | 二分类正确/错误，判别最终答案是否正确 | 数学推理 GSM8K |
| **PRM (Process RM)** | 步骤级分类 {-1, 0, 1}，评估推理每一步质量 | 数学推理 PRM800K |
| **Generative RM** | 用 LLM 的生成概率作为 reward（无需额外头） | 更灵活 |

**面试深挖点**：
- Q: 为什么 RM 通常只训练 1 epoch？
  A: RM 本身是过优化的源头之一。训练过久会过拟合到训练集偏好，降低泛化能力
- Q: ORM vs PRM 的核心区别？各自优劣？
  A: ORM 只看最终答案，简单但信号稀疏；PRM 看每一步，信号密集但标注成本高且存在 step boundary 问题
- Q: Bradley-Terry 模型假设了什么？有什么局限？
  A: 假设偏好是传递性的（A>B, B>C -> A>C），且偏好强度只取决于分数差。实际上人类偏好可能不传递、有上下文依赖
- Q: RM 的 reward head 为什么从 EOS token 取 hidden state 而不是用 mean pooling?
  A: EOS token 理论上聚合了整个序列的自回归信息。实践中两者都可行，EOS 是最常见选择（InstructGPT 传统）

### 2.3 阶段三：RL 训练（策略梯度）

**核心原理**：用 RM 打分作为奖励信号，通过策略梯度方法更新 LLM 的参数。

**标准 RLHF 目标**：
```
max_pi  E_{x~D, y~pi(y|x)}[r_theta(x,y)] - beta * D_KL(pi(y|x) || pi_ref(y|x))
```
- 第一项：最大化奖励模型打分
- 第二项：KL 惩罚，防止策略偏离参考模型太远（避免 over-optimization）
- beta：控制两者的权衡

**策略梯度核心公式**：
```
Delta_theta proportional to  Psi_t * grad_theta log pi_theta(a_t | s_t)
```
- `Psi_t > 0`：更新参数让 a_t 更可能出现
- `Psi_t < 0`：更新参数让 a_t 更不可能出现

---

## 5. 策略梯度 (Policy Gradients) 深度要点

### 5.1 四种主要算法对比

| 算法 | 核心特点 | 是否需要 Value Model | Advantage 估计 |
|------|---------|---------------------|----------------|
| **REINFORCE** | 最简单的策略梯度，用 return 作为 baseline | 否 | G_t - baseline |
| **RLOO** | REINFORCE Leave-One-Out，用其他样本均值做 baseline | 否 | r_i - mean(r_{j!=i}) |
| **PPO** | Clipped surrogate objective + value function + GAE | 是 | GAE(lambda) |
| **GRPO** | Group Relative Policy Optimization，组内相对比较 | 否 | 组内归一化 |
| **DAPO** | Decoupled Clip + Dynamic Sampling | 否 | 解耦上下 clip |

### 5.2 PPO vs GRPO 详细对比（核心面试题）

**PPO（Proximal Policy Optimization）**：
- 需要额外的 value model（和 policy 一样大，额外占用显存）
- 使用 GAE（Generalized Advantage Estimation）估计 advantage
- Clipped objective: `L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)`
- r_t = pi_new(a|s) / pi_old(a|s) 是 importance ratio
- 4个模型：policy + reference + reward + value

**GRPO（Group Relative Policy Optimization）**：
- 不需要 value model（节省显存和计算）
- 对同一个 prompt 生成一组（group）回复，用组内 reward 的均值和标准差做归一化
- Advantage: `A_i = (r_i - mean(r_group)) / std(r_group)`
- 3个模型：policy + reference + reward
- 特别适合推理任务（数学、代码）

### 5.3 关键实现细节

**KL 惩罚**：
- 在 reward 中减去 KL 惩罚：`r = r_model - beta * D_KL(pi || pi_ref)`
- KL 的近似实现（Monte Carlo 估计）：
  ```
  kl_approx = log(pi(a|s)) - log(pi_ref(a|s))
  ```
  对每个 token 计算然后求和

**代码核心循环**（来自 `code/policy_gradients/train.py`）：
```python
for replay_buffer in rollout_engine:
    # 1. 采样: 用当前 policy 对 prompts 生成 completions
    # 2. 打分: 用 RM 对 completions 打分
    # 3. 计算 advantage
    for exp in experience_sampler:
        log_probs = compute_log_probs(model, exp.sequence_ids, exp.attention_mask)
        loss = objective(log_probs=log_probs, experience=exp, values=values)
        scaled_loss = loss / batch_acc
        scaled_loss.backward()
    # 4. 更新参数
    clip_grad_norm_(params, max_norm=max_norm)
    optimizer.step()
```

**GRPO 配置示例**（`policy_gradients/configs/grpo.yaml`）：
```yaml
loss: grpo
model_name: Qwen/Qwen3-1.7B
clip_eps_lo: 0.2
clip_eps_hi: 0.2
beta: 0.0           # KL penalty (0=disabled)
lr: 5e-6
temperature: 0.6
prompts_per_step: 4
num_rollouts: 8      # 每个 prompt 生成 8 个回复
train_batch_size: 2
max_norm: 1.0        # gradient clipping
```

**面试深挖点**：
- Q: GRPO 的 num_rollouts 是什么意思？为什么设置为 8？
  A: 每个 prompt 生成 8 个回复组成一个 group，用于组内归一化估计 advantage。数量太小方差大，太大增加计算开销
- Q: 为什么 RLHF 中 gamma（折扣因子）通常设为 1？
  A: 因为优化单元是整个 completion 而不是单个 token，且生成是有限 horizon 的，不需要折扣
- Q: beta 设为 0 会怎样？
  A: 关闭 KL 惩罚，模型可能快速偏离参考分布，导致生成退化（乱码、重复等）
- Q: 为什么需要 clip_grad_norm？
  A: RL 训练中 reward 信号可能非常 noisy，梯度爆炸风险高。clip 确保训练稳定

---

## 6. 直接对齐算法 (DPO 及变体) 深度要点

### 6.1 DPO 核心思想

**核心洞察**：你的语言模型本身就是一个隐式奖励模型。

DPO 从 RLHF 目标出发，通过数学推导得到闭式最优策略：
```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
```

反解出隐式奖励：
```
r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log(Z(x))
```

代入 Bradley-Terry 模型后，Z(x) 被消掉，得到 DPO loss：
```
L_DPO = -E[log sigma(beta * (log pi(y_c|x)/pi_ref(y_c|x) - log pi(y_r|x)/pi_ref(y_r|x)))]
```

**直觉理解**：
- 提高 chosen 的相对概率（相对于 reference model）
- 降低 rejected 的相对概率（相对于 reference model）
- beta 控制 KL 约束强度

### 6.2 DPO 的梯度分析

```
grad_theta L_DPO = -beta * E[w * (grad log pi(y_c|x) - grad log pi(y_r|x))]
```
其中 `w = sigma(r(x,y_r) - r(x,y_c))`

- 当 RM 对 chosen 和 rejected 的排序**正确**时，w 小 -> 小更新
- 当 RM 排序**错误**时，w 大 -> 大更新
- 本质上 DPO 在拟合一个**隐式奖励模型**

### 6.3 DPO 变体对比

| 算法 | 核心区别 | 优势 |
|------|---------|------|
| **DPO** | 标准 BT loss + reference model | 简单，广泛验证 |
| **cDPO** | 带 label smoothing 的 DPO | 更鲁棒的噪声标签 |
| **IPO** | 用 squared loss 替代 sigmoid | 不依赖 BT 假设，对噪声更鲁棒 |
| **SimPO** | 长度归一化 + 无 reference model | 更简单，不需要加载 ref model |
| **ORPO** | 合并 SFT + 偏好优化 | 单阶段训练 |
| **KTO** | 只需 thumbs up/down 标签（非成对） | 数据收集更简单 |
| **APO-Zero** | chosen 上升 + rejected 下降 | anchored 更新 |
| **APO-Down** | 两者都下降，但 rejected 降更多 | 更保守 |

### 6.4 RL 方法 vs DPO 方法（核心对比）

| 维度 | RL (PPO/GRPO) | DPO |
|------|---------------|-----|
| **是否 online** | 是（当前模型生成） | 否（离线数据） |
| **是否需要 RM** | 是 | 否（隐式） |
| **计算成本** | 高（4个模型 + 生成） | 低（2个模型） |
| **实现复杂度** | 高 | 低 |
| **beta 含义** | KL 惩罚系数 | 隐式温度参数 |
| **过优化风险** | 有（reward hacking） | 有但更可控 |
| **性能上限** | 更高（online 探索） | 略低 |

**面试深挖点**：
- Q: DPO 为什么说是"你的 LM 就是 RM"？
  A: DPO 的隐式奖励 r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))，直接从 policy 的 log prob 中得到
- Q: DPO 的 beta 和 RLHF 的 beta 有什么区别？
  A: 在 RLHF 中 beta 是 KL 惩罚权重，控制探索-利用；在 DPO 中 beta 类似温度，控制偏好学习的强度
- Q: 为什么 DPO 需要非常低的学习率？
  A: DPO 在离线数据上做梯度上升，数据分布和当前策略不完全匹配，太大的 LR 会导致灾难性更新
- Q: DPO 本质上在拟合什么？
  A: 本质上在拟合一个隐式奖励模型（Bradley-Terry 形式），然后同时优化策略

---

## 7. 拒绝采样 (Rejection Sampling) 深度要点

### 7.1 流程

1. **生成**：对每个 prompt 生成 N 个 completions
2. **打分**：用 RM 对所有 completions 打分
3. **选择**：选择 top completions（per-prompt top-1 或 overall top-K）
4. **SFT**：在选中的 completions 上做 SFT

### 7.2 两种选择策略

- **Top Per Prompt**：每个 prompt 取最高分的 1 个 completion
- **Top K Overall**：从所有 prompt-completion 对中取最高分的 K 个

### 7.3 为什么有效

拒绝采样等价于一种特殊的 RL：它使用奖励模型做推理时的 "重采样"，不需要在线更新参数。从统计角度看，它是从一个更容易采样的分布中按条件筛选样本。

**面试深挖点**：
- Q: 拒绝采样和 Best-of-N 的关系？
  A: Best-of-N 是拒绝采样的推理时版本（inference-time），选择 top-1 回复给用户。拒绝采样是训练时版本，选 top completions 做 SFT
- Q: 拒绝采样为什么在 RLHF 流水线中放在哪个位置？
  A: 通常在 SFT 之后、RL 之前，或者在 RL 之后做 refinement。它是 RL 方法的一种简单替代

---

## 8. 推理模型与 RLVR

### 8.1 核心思想

**RLVR (Reinforcement Learning with Verifiable Rewards)**：用可验证的奖励函数替代 RM。

```
数学: extracted_answer == correct_answer -> reward = 1
代码: all_tests_pass -> reward = 1
```

### 8.2 Thinking Models 的训练

核心循环：
1. 对多个问题采样多个答案
2. 对正确答案做梯度上升
3. 重复，反复访问相同数据

**关键洞察**：即使是简单的方法（反复用正确答案训练），也能让模型学习到推理行为，并且泛化到没见过的问题。

### 8.3 RLHF vs RLVR

| 维度 | RLHF | RLVR |
|------|------|------|
| 奖励来源 | 学习的 RM（代理目标） | 验证函数（真值） |
| 领域 | 主观（风格、安全性） | 客观（数学正确性） |
| 过优化风险 | 高 | 低（目标明确） |
| 代表模型 | ChatGPT, Claude | o1, DeepSeek R1 |

### 8.4 推理训练的起源

**Yann LeCun 的蛋糕比喻**：
- 蛋糕体 = 无监督学习（预训练）
- 糖衣 = 有监督学习（SFT）
- 樱桃 = 强化学习（RLHF/RLVR）

**面试深挖点**：
- Q: RLVR 和 RLHF 有什么本质区别？
  A: RLVR 的奖励是确定性的（答案对不对），不需要训练 RM；RLHF 的奖励是学习的（代理目标），存在 over-optimization 问题
- Q: 为什么推理模型需要很长的 thinking tokens？
  A: 这是 inference-time scaling 的体现。更长的思考链让模型可以 "搜索" 更多解题路径。RL 训练自然地学会了生成更长的思考
- Q: 推理模型训练中，模型是怎么学会思考的？
  A: 通过反复访问相同问题，RL 增加了导向正确答案的行为概率。模型学到哪些 "思考模式" 与正确答案相关

---

## 9. 工具调用与 Agent

### 9.1 三种相关概念

- **Tool Use**：模型输出结构化请求（工具名+参数），编排器执行工具，结果追加到上下文
- **Function Calling**：工具调用，参数必须符合声明的 schema（通常是 JSON Schema）
- **Code Execution**：特殊工具调用，工具是代码解释器

### 9.2 工具调用的训练数据格式

```
<system>
You are a function-calling AI model...
<functions>[JSON schema]</functions>
</system>
<user>...</user>
```

模型生成工具调用 token -> 编排器执行 -> 结果以特殊 token 插入 -> 模型继续生成

### 9.3 编排循环

```python
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        break  # 模型决定不调用工具，直接回复
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append({"role": "tool", "content": result})
```

### 9.4 Agent 与 RLHF 的关系

工具调用是 Agent 系统的基础能力。RLHF 可以用于：
- 提高工具调用的准确性（选对工具、参数正确）
- 决定何时调用工具 vs 直接回答
- 多步工具调用的规划能力

**面试深挖点**：
- Q: 如何训练模型的工具调用能力？
  A: SFT 阶段用工具调用数据训练格式，RLHF 阶段用工具调用正确性作为 reward signal 优化
- Q: Agent 系统中 RLHF 的作用是什么？
  A: 优化 agent 的决策策略（何时调用工具、调用哪个、如何组合结果）
- Q: 工具调用的评估指标有哪些？
  A: exact-match（工具名+参数）、schema 合法性、pass^k（一致性）、端到端任务完成率

---

## 10. 过优化与正则化

### 10.1 过优化的两种表现

1. **Reward Over-optimization**：训练 RM score 持续上升，但实际质量下降
2. **Qualitative Degradation**：模型变得冗长、阿谀奉承（sycophancy）、刻板

### 10.2 Goodhart's Law

> 当一个指标成为目标时，它就不再是一个好指标。

RM 是代理目标（proxy objective），强优化器（RL）会找到 hack RM 的方式，而不是真正提升质量。

### 10.3 常见症状

- 常见短语："As an AI language model..." "Certainly!..."
- 重复性、不提供信息的回答
- 讨好用户：自我怀疑、阿谀奉承、过度道歉
- 过度拒绝（over-refusal）：误杀正常请求

### 10.4 正则化方法

**KL 惩罚**（最核心）：
```
r = r_theta - lambda_KL * D_KL(pi_RL || pi_ref)
```

实现方式：计算当前策略和参考模型在生成 token 上的 log-prob 差值：
```python
kl_approx = seq_logprob - ref_seq_logprob
reward = rm_reward - beta * kl_approx
```

**其他方法**：
- Reward Model Ensemble（多个 RM 取平均/最小）
- 换优化器（如解耦 clip 的 DAPO）
- DPO 的隐式 KL（通过数据约束）

### 10.5 DPO vs RL 的过优化

DPO 也有过优化问题，但更可控：
- DPO 的 beta 直接控制 KL 距离（因为解是闭式的）
- RL 的 KL 取决于优化过程，更难精确控制

**面试深挖点**：
- Q: 为什么 KL 惩罚能防止过优化？
  A: 限制模型不偏离 reference model 太远，避免进入 reward model 评估不准确的 OOD 区域
- Q: beta 设多大合适？
  A: 没有通用答案。beta 太大 -> 学不到东西；beta 太小 -> 过优化。需要 sweep 找最优值
- Q: 什么情况下可以不用 KL 惩罚？
  A: RLVR 场景（验证函数是确定性的，不容易被 hack）

---

## 11. 偏好数据与合成数据

### 11.1 偏好数据的关键概念

**On-policy 数据**：从当前模型（或其近亲 checkpoint）收集的偏好数据。研究表明 on-policy 数据对 RLHF 训练至关重要。

**数据收集方式**：
- **Rankings**：排序（相对），更常见
- **Ratings**：评分（绝对），信息更丰富但噪声大
- **Thumbs Up/Down**：二值反馈，最简单

### 11.2 合成数据的三大角色

1. **SFT 数据**：从强模型 distillation（如 GPT-4 -> 小模型），已大幅替代人工数据
2. **Preference 数据**：AI 反馈（RLAIF）或 Constitutional AI，前沿模型仍在混用人工+合成
3. **评估**：LLM-as-a-Judge 扩展评估规模，但 ground truth 仍需人工

### 11.3 合成数据的核心权衡

- **优势**：成本低、规模大、可迭代、一致性高
- **风险**：Model Collapse（反复自训练导致分布退化）
- **缓解**：混入真实数据、多样化 teacher 模型、去重、质量过滤

### 11.4 Distillation（蒸馏）

```
Teacher Model (强) -> 生成 completions -> 训练 Student Model (弱)
```

**两种形式**：
1. 作为通用数据引擎（SFT、偏好数据、验证信号）
2. 专门转移特定技能（数学推理、代码）

**面试深挖点**：
- Q: 为什么 on-policy 数据重要？
  A: 不同模型有不同的生成分布，off-policy 数据的 completions 风格可能与当前模型差距大，导致优化不稳定
- Q: 合成偏好数据能完全替代人工数据吗？
  A: 学术界研究显示效果接近，但前沿模型仍认为人工数据是竞争壁垒（尤其在最难的问题上）
- Q: 如何避免 model collapse？
  A: 混合真实数据、使用多个 teacher 模型、deduplication、强质量过滤

---

## 12. 评估体系

### 12.1 评估演变三个阶段

1. **早期 Chat 阶段**：MT-Bench, AlpacaEval, Arena-Hard（LLM-as-Judge）
2. **多技能时代**：MMLU, BigBenchHard, MATH, GSM8K, HumanEval, IFEval
3. **推理与工具时代**：GPQA Diamond, SWE-Bench, LiveCodeBench, AIME

### 12.2 评估中的 Prompting 敏感性

- 不同的 prompt 格式可以让模型表现从 60% 掉到接近 0%
- Few-shot vs zero-shot、log-likelihood vs 生成、chain-of-thought 都有巨大影响
- **教训**：强模型更容易被打破（prompt 格式不对）而非变得更强

### 12.3 评估方差问题

| 类别 | 标准差 | 代表 benchmark |
|------|--------|----------------|
| 高方差 | >1.0 | GPQA, AlpacaEval 3, IFEval |
| 稳定 | 0.4-0.6 | ZebraLogic, AIME, HumanEval+ |
| 非常稳定 | <0.3 | MATH, MMLU, PopQA |

**面试深挖点**：
- Q: 如何处理评估方差？
  A: 多次运行取平均（如 AIME Avg@32）、用更稳定的 benchmark、控制 sampling temperature
- Q: 为什么推理模型的评估更难？
  A: 需要 temperature > 0 采样，生成长度长，不同 batch size / TP 设置都影响结果

---

## 13. Character Training 与产品化

### 13.1 Character Training

通过后训练塑造模型的"人格"特征（好奇心、开放性、深思熟虑等）。

**方法**：
1. 定义特征 -> 生成相关 queries -> 模型生成回复 -> 按特征排序 -> 训练
2. 类似 Constitutional AI，但目标是人格而非安全性
3. 重度依赖合成数据，但需要人工 "艺术家触觉" 检查

### 13.2 Persona Vectors

**核心思想**：人格特征对应模型 residual stream 中的线性方向。

提取方法：对比正向/反向 prompt 的 activation 差异，得到 persona vector。推理时通过加减该向量控制人格。

```
v_layer = mean(positive_activations) - mean(negative_activations)
inference: residual += alpha * v_layer  # alpha > 0 增强, < 0 抑制
```

**面试深挖点**：
- Q: Character Training 和 RLHF 的关系？
  A: 使用相同的方法（SFT + RLHF/DPO），但数据更聚焦于特定人格特征
- Q: Persona Vectors 和 Activation Steering 的关系？
  A: Persona Vectors 是 Activation Steering 的具体应用，用对比 prompt 提取人格方向

---

## 14. 代码实现细节速查

### 14.1 SFT 训练核心 (`code/instruction_tuning/train.py`)

```python
# 核心循环
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = compute_loss(model, batch)  # standard CE loss with prompt masking
        scaled = loss / accum
        scaled.backward()
        # 每 accum 步更新一次
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
```

**关键配置**：AdamW, gradient accumulation, linear warmup + decay LR

### 14.2 奖励模型训练核心 (`code/reward_models/train_preference_rm.py`)

```python
class PreferenceRewardModel(BaseRewardModel):
    def __init__(self, model_id):
        super().__init__(model_id, head_dim=1)  # 线性头输出1维
    
    def get_reward(self, input_ids, attention_mask):
        hidden = self.get_hidden_states(input_ids, attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[batch_indices, seq_lengths]
        return self.head(last_hidden).squeeze(-1)

# 训练
loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

### 14.3 DPO 训练核心 (`code/direct_alignment/train.py`)

```python
# Forward pass: 分别计算 policy 和 ref 的 log probs
policy_chosen_logps = forward_pass(policy_model, batch.chosen_input_ids, ...)
policy_rejected_logps = forward_pass(policy_model, batch.rejected_input_ids, ...)
ref_chosen_logps = forward_pass(ref_model, batch.chosen_input_ids, ...)
ref_rejected_logps = forward_pass(ref_model, batch.rejected_input_ids, ...)

# Loss
loss = loss_fn(policy_chosen_logps, policy_rejected_logps, 
               ref_chosen_logps, ref_rejected_logps)
```

### 14.4 策略梯度训练核心 (`code/policy_gradients/train.py`)

```python
for replay_buffer in rollout_engine:
    # Rollout: 生成 completions + 打分
    for exp in experience_sampler:
        log_probs = compute_log_probs(model, exp.sequence_ids, exp.attention_mask)
        values = compute_values(val_model, exp.sequence_ids, exp.attention_mask)
        loss = objective(log_probs=log_probs, experience=exp, values=values)
        scaled_loss.backward()
    clip_grad_norm_(params, max_norm=max_norm)
    optimizer.step()
```

### 14.5 内存需求参考

| 训练类型 | 模型 | GPU 内存 |
|---------|------|---------|
| Policy gradients | Qwen3-1.7B | ~16GB |
| Reward models | Qwen3-0.6B | ~8-16GB |
| Reward models | Qwen3-1.7B | ~16-20GB |
| SFT (full finetune) | 0.6B | ~4-6GB |
| SFT (full finetune) | 1.7B | ~10-15GB |
| SFT (full finetune) | 3B | ~20-25GB |

---

## 15. 面试深挖问题集

### 第一层：概念理解（初级）

1. **RLHF 的三个阶段分别是什么？各阶段的目标？**
   - SFT：教格式 -> RM：学偏好 -> RL：优化策略

2. **DPO 和 RLHF 的核心区别是什么？**
   - DPO 不需要训练显式 RM，不在线生成，直接从偏好数据优化策略

3. **什么是 Chat Template？为什么重要？**
   - 定义模型输入输出格式的 token 序列模式，确保模型正确识别角色

4. **Bradley-Terry 模型是什么？**
   - 偏好概率模型 P(i>j) = sigmoid(r_i - r_j)

5. **KL 惩罚的作用是什么？**
   - 防止策略偏离参考模型太远，避免 over-optimization

### 第二层：实现细节（中级）

6. **SFT 中的 Prompt Masking 具体是怎么做的？**
   - 将 prompt tokens 的 label 设为 -100（PyTorch CrossEntropyLoss 会忽略），只在 assistant 回复 tokens 上计算 loss

7. **PPO 需要几个模型？各自的作用？**
   - 4个：Policy（被训练的模型）、Reference（计算 KL）、Reward（打分）、Value（估计 state value，用于 advantage 计算）

8. **GRPO 为什么不需要 Value Model？**
   - 用同一 prompt 的多个回复的组内 reward 统计量（均值、标准差）来估计 advantage

9. **DPO 的隐式奖励公式推导？**
   - 从 RLHF 目标的闭式解 pi* = (1/Z) * pi_ref * exp(r/beta) 反解 r = beta * log(pi/pi_ref) + beta*log(Z)

10. **拒绝采样的 Top-Per-Prompt 和 Top-K-Overall 各有什么优劣？**
    - Top-Per-Prompt：每 prompt 都有样本，多样性好；Top-K-Overall：可能集中在少数 prompt 上但总体质量最高

### 第三层：深度分析（高级/实习深挖）

11. **为什么 DPO 被认为是 "你的 LM 就是 RM"？这对训练有什么影响？**
    - DPO 的隐式奖励直接从 policy log prob 得到，policy 同时是 generator 和 scorer
    - 影响：beta 的选择变得更关键，因为没有独立 RM 来校准

12. **在线 RL（PPO/GRPO）vs 离线（DPO）的 trade-off 分析？**
    - Online：on-policy 数据 -> 更准的梯度 -> 更高的上限，但计算成本高
    - Offline：数据固定 -> 可能 off-policy -> 有上限，但实现简单
    - 趋势：工业界越来越重视 online RL（如 RLVR）

13. **Over-optimization 的根本原因是什么？如何量化？**
    - 根因：代理目标（RM）和真实目标（用户满意度）不完全对齐
    - 量化：监控 KL 距离 vs 下游评估的帕累托曲线；训练 RM score vs test RM score 的交叉点

14. **Process Reward Model (PRM) 的核心挑战是什么？**
    - Step boundary 定义困难（什么算"一步"？）
    - 标注成本高（需要步骤级标注）
    - 但信号更密集，在数学推理中效果更好

15. **如果你要设计一个 Agent 系统的后训练流水线，你会怎么做？**
    - SFT 阶段：用工具调用数据训练格式
    - RM 阶段：用工具调用正确性 + 最终任务完成率构建偏好数据
    - RL 阶段：用 RM 打分 + 可验证奖励（如代码测试通过率）做 RL
    - 关键：on-policy 数据收集、KL 控制、评估覆盖多种工具组合

16. **Constitutional AI 的核心思路？它和 RLHF 的关系？**
    - CAI 用一组"宪法"原则让 AI 自我批评和修订，生成偏好数据
    - 本质是用 AI 反馈替代人类反馈，降低成本
    - 与 RLHF 共享相同的下游训练流程（SFT + 偏好微调）

17. **RLVR 和 RLHF 在过优化风险上有什么不同？**
    - RLVR：奖励是确定性的（对/错），不容易被 hack
    - RLHF：奖励是学习的代理目标，容易被 hack
    - 所以 RLVR 可以用更少的正则化、更多的训练步数

18. **Character Training 的数据是怎么构造的？和普通 SFT 数据有什么区别？**
    - 用强模型按人格特征生成 query 和 ranked responses
    - 区别：目标更聚焦（特定人格维度），数据量通常较小但需要更精细的质量控制

---

## 16. 自我介绍框架建议

### 面试开场白模板（2-3分钟）

> 面试官您好，我是 [名字]，[学校] [专业] 的在读硕士。
>
> 我对 LLM 后训练方向有深入的学习和实践经验。我系统研读了 Nathan Lambert 的 RLHF Book（rlhfbook.com），这是一个涵盖了从指令微调、奖励模型、策略梯度到直接对齐算法的完整后训练知识体系。
>
> 在技术理解上，我对 RLHF 的完整流水线有深入理解：
> - **SFT 阶段**：理解 chat template、prompt masking、学习率选择等关键实现细节
> - **奖励模型**：掌握 Bradley-Terry 模型的数学推导，了解 ORM、PRM 等变体的区别和适用场景
> - **策略梯度**：理解 PPO/GRPO/REINFORCE 等算法的数学原理和实现差异，特别是 GRPO 如何通过组内归一化避免使用 value model
> - **直接对齐**：能够推导 DPO 的完整数学过程，理解其隐式奖励模型的本质
>
> 在实践上，我运行了该项目的参考代码，包括：
> - 用 GRPO 在 spell_backward 任务上训练策略模型
> - 用 DPO 在 UltraFeedback 上做偏好对齐
> - 训练 Bradley-Terry 奖励模型
>
> 我还关注到后训练领域的最新发展，包括推理模型（RLVR）、工具调用训练、Character Training 等前沿方向。
>
> 我希望能将这些知识应用到实际的 LLM 后训练或 Agent 系统开发中。

### 准备策略

1. **通读本文档**：确保每个知识点都能流利解释
2. **动手跑代码**：至少运行一次 SFT + GRPO + DPO 的完整流程
3. **准备数学推导**：Bradley-Terry -> RM loss、DPO 推导、策略梯度定理
4. **关注最新进展**：DeepSeek R1、GRPO、DAPO 等 2025 年新方法
5. **准备项目经验**：即使是学习性质的实验，也要能描述"做了什么、观察到了什么、为什么"

---

*本文档基于 RLHF Book 项目所有 17 章 + 3 附录 + 5 个代码模块 + 课程讲义系统整理。*
*项目地址: https://rlhfbook.com | 代码: https://github.com/natolambert/rlhf-book*
*生成日期: 2026-06-27*
