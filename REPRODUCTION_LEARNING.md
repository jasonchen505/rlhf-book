# 复现过程增量学习笔记

> 相比 INTERVIEW_PREP.md（知识体系）和 INTERVIEW_DEEP_DIVE.md（面试应答），本文档记录在实际复现过程中新发现的、容易忽略的、或与直觉相悖的知识点。
> 随复现进度持续更新。

---

## 目录

- [一、环境与工程细节](#一环境与工程细节)
- [二、SFT 阶段增量知识](#二sft-阶段增量知识)
- [三、Reward Model 增量知识](#三reward-model-增量知识)
- [四、Policy Gradient 增量知识](#四policy-gradient-增量知识)
- [五、DPO 及变体增量知识](#五dpo-及变体增量知识)
- [六、Rejection Sampling 增量知识](#六rejection-sampling-增量知识)
- [七、跨模块全局洞察](#七跨模块全局洞察)
- [八、直觉 vs 现实的偏差](#八直觉-vs-现实的偏差)

---

## 一、环境与工程细节

### 1.1 代码是纯单 GPU 设计

**发现**：grep 搜遍整个 code/ 目录，没有任何 `DistributedDataParallel`、`FSDP`、`torch.cuda.device_count` 的使用。所有训练脚本只用 `model_device_id` 指定单卡。

**这意味着**：
- 8 张 4090 的用法是**并行跑不同实验**，而非分布式训练一个实验
- 想跑 7B+ 模型需要自己加 FSDP 或 DeepSpeed
- 项目的教育目的优先于工业级扩展

### 1.2 Flash Attention 在 4090 上的实际表现

**代码处理**（来自 `direct_alignment/train.py`）：
```python
def get_attn_implementation() -> str:
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return "sdpa"
```

**新知识点**：
- 4090 (x86_64) 支持 flash_attention_2，但需要手动安装 `flash-attn`
- 如果不装，自动 fallback 到 PyTorch SDPA（功能等价，速度稍慢）
- DGX Spark (aarch64/Blackwell) 不支持 flash-attn，SDPA 反而更快

### 1.3 uv 包管理器

**之前不知道的**：项目用 `uv` 而非 `pip`。uv 是 Astral 出品的超快 Python 包管理器。

```bash
# 不是 pip install -e . 而是
cd code/ && uv sync

# 不是 python train.py 而是
uv run python -m instruction_tuning.train --config xxx.yaml
```

**为什么要用 uv**：锁文件确保所有人的依赖版本一致。`uv sync` 比 `pip install` 快 10-100x。

### 1.4 模型下载的陷阱

**发现**：Qwen3 系列模型需要 `trust_remote_code=True`，否则 tokenizer 加载失败。OLMo-2 的 base model 没有 chat_template，需要从 SFT 版本复制：

```yaml
# sft_olmo2_1b.yaml 中
model_name: allenai/OLMo-2-0425-1B          # base model
chat_template_source: allenai/OLMo-2-0425-1B-SFT  # chat template 来自 SFT 版本
```

---

## 二、SFT 阶段增量知识

### 2.1 in-loop sample generation 是关键调试工具

**之前理解**：SFT 就是训 loss。
**实际发现**：`sample_every: 50` 每 50 步在固定 prompt pool 上生成回复，直接观察 base->assistant 的转变过程。

**为什么重要**：
- loss 下降不等于模型学会了回答——可能只是学会了短回复
- in-loop samples 能看到格式是否正确、是否停止生成、是否有幻觉
- 这是 SFT 阶段最直观的 sanity check

### 2.2 prompt masking 的实现细节

**代码中**（`instruction_tuning/utils.py` 的 `compute_loss`）：
- prompt tokens 的 label 设为 -100
- PyTorch `CrossEntropyLoss` 会自动忽略 -100 的位置
- 只有 assistant 回复部分参与 loss 计算

**容易出错的点**：
- 多轮对话中，中间的 assistant 回复是否参与 loss？两种策略（final-turn only vs mask user only）效果不同
- 如果 chat template 格式不对，prompt 和 response 的边界识别错误，masking 就错了

### 2.3 OLMo-2 的特殊 tokenizer

**发现**：OLMo-2 base model 的 tokenizer 没有 `chat_template`。代码通过 `chat_template_source` 从 SFT 版本复制模板。

**启示**：不同模型的 tokenizer 差异巨大。做 SFT 时必须确认 chat template 是否正确设置。

---

## 三、Reward Model 增量知识

### 3.1 RM 训练只需要 1 epoch 的深层原因

**之前理解**：怕过拟合。
**更深层的原因**：RM 本身是 RLHF 过优化的源头之一。如果 RM 过拟合到训练数据，它在 OOD 区域的评分就更不可信，RL 优化器就更容易 hack 它。1 epoch 是在"学到偏好"和"保持泛化"之间的平衡。

### 3.2 EOS token 选择的实现细节

**代码**（`reward_models/train_preference_rm.py`）：
```python
def get_reward(self, input_ids, attention_mask):
    hidden = self.get_hidden_states(input_ids, attention_mask)
    seq_lengths = attention_mask.sum(dim=1) - 1  # 最后一个非 padding token 的位置
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    last_hidden = hidden[batch_indices, seq_lengths]
    reward = self.head(last_hidden).squeeze(-1)
    return reward
```

**关键细节**：不是用 `input_ids[:, -1]`，而是用 `attention_mask.sum()` 找到最后一个真实 token 的位置。因为 batch 中不同序列长度不同，padding 的位置不能用。

### 3.3 ORM vs PRM 的标注成本差异

**之前理解**：PRM 信号更密集所以更好。
**实际发现**：PRM 需要步骤级标注（PRM800K 数据集），标注成本远高于 ORM（只需要最终答案对/错）。而且"什么算一步"没有标准定义，不同标注者的 step boundary 不一致。

### 3.4 Preference RM 的 loss 两种等价形式

```python
# 形式 1：log-sigmoid（常用）
loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

# 形式 2：softplus（数学等价）
loss = F.softplus(rewards_rejected - rewards_chosen).mean()
```

**为什么两种都出现**：不同论文/框架的习惯不同。InstructGPT 用形式 1，Anthropic 用形式 2。

---

## 四、Policy Gradient 增量知识

### 4.1 spell_backward 任务的设计巧思

**发现**：默认任务是 `spell_backward`（把单词倒着拼写）。这不是随意选的：
1. **可验证**：答案对不对一目了然（verifiable reward）
2. **难度可控**：`min_word_len: 3, max_word_len: 10` 控制难度
3. **格式要求**：需要遵循特定输出格式（format_weight 控制）
4. **快速收敛**：1.7B 模型通常几百步就能学会

**启示**：选对实验任务比调参更重要。好的实验任务应该是可验证、可控难度、快速反馈的。

### 4.2 GRPO 的 num_rollouts 不是 batch size

**之前混淆的**：以为 num_rollouts 是每个 batch 的样本数。
**实际**：
- `prompts_per_step: 4`：每个 step 处理 4 个 prompt
- `num_rollouts: 8`：每个 prompt 生成 8 个 completion
- 一个 step 生成 4 * 8 = 32 个 completion
- `train_batch_size: 2`：训练时每个 batch 2 个 completion
- `batch_acc: 4`：gradient accumulation 4 步

所以实际有效 batch size = 2 * 4 = 8 个 completion per optimizer step。

### 4.3 PPO 需要 4 个模型的显存规划

**PPO 配置中的特殊字段**：
```yaml
val_model_device_id: 0  # value model 和 policy 放同一张卡
```

**显存计算**：
- Policy: ~3.4GB
- Reference: ~3.4GB  
- Value: ~3.4GB（和 policy 一样大）
- Reward: ~1.2GB
- **总计**：~11.4GB 模型参数 + optimizer states + activations ≈ 16-20GB

**这意味着**：PPO 在 1.7B 模型上可以单卡跑，但如果模型再大就需要分卡。

### 4.4 KL estimator 的三种选择

**配置中**：`kl_estimator: kl3`

**三种 KL 估计器**：
- `kl1`：`log(pi/pi_ref)` 的均值
- `kl2`：`pi/pi_ref - log(pi/pi_ref) - 1`（更稳定的近似）
- `kl3`：`0.5 * (log(pi/pi_ref))^2`（二阶近似）

**为什么 kl3 是默认**：在数值上更稳定，特别是当 pi/pi_ref 接近 1 时。

### 4.5 Loss 聚合策略的三个选择

**之前没注意到的**（来自书中详细讨论）：

1. **Per-sequence normalization**（GRPO 默认）：每个序列贡献相等
2. **Per-token normalization**（DAPO）：每个 token 贡献相等，长序列权重更大
3. **Fixed-length normalization**（Dr. GRPO）：用 L_max 归一化

**为什么重要**：不同策略会导致模型对长/短回复的偏好不同。Per-sequence 可能导致模型倾向短回复（因为短回复的 per-token loss 更大）。

---

## 五、DPO 及变体增量知识

### 5.1 DPO 需要极低学习率的原因

**之前理解**：DPO 比 RL 简单所以 LR 要低。
**更深层的原因**：DPO 是在离线数据上做梯度上升。数据中的 completions 来自旧策略，和当前策略可能不匹配。太大的 LR 会导致策略突变，进入训练数据覆盖不到的区域。

**实际经验值**：DPO 的 LR 在 1e-7 到 5e-6 之间，远低于 SFT 的 1e-5 到 5e-5。

### 5.2 SimPO 不需要 reference model 的代价

**代码中**（`direct_alignment/README.md`）：
> SimPO requires lower learning rates than DPO (3e-7 to 1e-6). A large learning rate (e.g., 1e-5) can significantly degrade performance.

**代价**：
- 没有 reference model 的 KL 约束，更需要低 LR 来稳定训练
- beta 值范围不同（DPO: 0.1-0.5, SimPO: 2.0-2.5）
- 对 LR 更敏感，调参空间更小

### 5.3 ORPO 的数值问题

**代码注释中的关键发现**：
> ORPO uses average log-probabilities, not summed. Summed log-probs on long responses become large-magnitude negatives, which makes the ORPO log-odds term numerically extreme.

**之前不知道的**：ORPO 的 log-odds ratio 在长序列上会数值爆炸，必须用 average log-prob 而非 sum。

### 5.4 DPO-Norm 的 beta 范围完全不同

**发现**：DPO-Norm（DPO + SimPO 的长度归一化但保留 ref model）的 beta 需要 2.0-5.0，和标准 DPO 的 0.1-0.5 差一个数量级。

**原因**：DPO-Norm 用 per-token average log-ratio，数值比 sum log-ratio 小很多，所以 beta 需要更大来补偿。

---

## 六、Rejection Sampling 增量知识

### 6.1 RS 的两个阶段可以分开跑

**代码结构**：
```
rejection_sampling/preprocess.py  # Stage 1: 生成 + 打分（慢）
rejection_sampling/train.py       # Stage 2: 选 top + SFT（快）
```

**工程细节**：preprocess 的结果可以缓存，多次 train 实验可以复用同一批 rollout 数据。

### 6.2 Top-Per-Prompt vs Top-K-Overall 的实际效果差异

**代码 README 中的结论**：
> On the reference 1k-train / 200-test GSM8K slice, top_k_overall beat its matched random baseline, while top_per_prompt and random_per_prompt were effectively tied.

**为什么**：Top-per-prompt 每个 prompt 只选 1 个，如果 prompt 都很简单，选出来的不一定比随机好多少。Top-K overall 可以集中在"RM 最有信心的"样本上。

### 6.3 RS 训练中需要对比 random baseline

**重要性**：RS 的效果必须和 random selection 对比。如果 top selection 不如 random，说明 RM 的选择能力不行（RM 质量差或者任务太简单）。

---

## 七、跨模块全局洞察

### 7.1 后训练的"数据 > 算法"定律

**从代码中观察到**：所有模块的默认数据集都经过精心选择：
- SFT: No Robots（人工标注，9.5K 高质量）
- RM: UltraFeedback（AI 反馈，清洗过的）
- Policy Gradients: spell_backward（程序化生成，可控）
- DPO: UltraFeedback binarized
- RS: GSM8K（数学，可验证）

**启示**：换数据集比换算法的影响通常更大。

### 7.2 每个算法的"默认任务"选择逻辑

| 模块 | 默认任务 | 选择原因 |
|------|---------|---------|
| SFT | No Robots | 人工标注，格式标准 |
| RM | UltraFeedback | 成对偏好数据 |
| Policy Gradients | spell_backward | 可验证、快速收敛 |
| DPO | UltraFeedback binarized | 标准偏好数据集 |
| RS | GSM8K | 数学可验证 |

**共同点**：都选了"有明确正确答案"或"有明确偏好方向"的任务。主观任务（如写诗好坏）更难验证。

### 7.3 评估方差是被严重低估的问题

**之前忽略的**：不同 benchmark 的方差差异巨大（GPQA std=1.48 vs MMLU std=0.22）。

**实际含义**：
- GPQA 上提升 2 分可能只是噪声
- MMLU 上提升 2 分是真实的改进
- 推理模型的评估需要多次运行取平均（AIME Avg@32）

**代码中的处理**：没有内置评估方差管理。需要自己多次运行。

### 7.4 Gradient Checkpointing 的通用性

**发现**：所有训练模块都默认开启 `gradient_checkpointing: true`。

**为什么**：后训练的 batch size 通常较小，但 sequence length 较长（2048+），activation 内存是瓶颈。gradient checkpointing 用时间换空间，减少 ~30-40% activation 内存。

---

## 八、直觉 vs 现实的偏差

### 8.1 "更多数据一定更好" -- 错

**SFT 的实践**：No Robots 只有 9.5K 样本，3 epochs，就能让 base model 学会回答格式。LIMA 论文用 1000 条就能达到不错的 chat 效果。

**真正重要的**：数据质量 > 数据数量。一条格式错误的数据可能比 100 条好数据更有害。

### 8.2 "DPO 比 PPO 简单所以效果差" -- 不完全对

**代码验证的结果**：在 1B 模型上，DPO 和 PPO 在 spell_backward 上效果相当。DPO 的上限取决于偏好数据的质量和 on-policy 程度。

**真正有差距的场景**：
- 推理任务（数学、代码）：online RL 显著优于 DPO
- 大规模模型：PPO 的上限更高但成本也高得多

### 8.3 "KL 惩罚越大越安全" -- 过于简化

**代码中的实际发现**：
- `beta: 0.0`（无 KL）在某些任务上也能工作（spell_backward 简单任务）
- `beta: 0.1` 在 DPO 中是标准值，但在 GRPO 中 0.0 是默认
- RLVR 任务通常关闭 KL（因为奖励是确定性的，不容易被 hack）

**关键洞察**：KL 惩罚的需求取决于奖励信号的可靠性。奖励越"软"（学习的 RM），越需要 KL；奖励越"硬"（可验证的），KL 越不重要。

### 8.4 "RL 训练一定很慢" -- 取决于任务

**spell_backward 任务**：GRPO 在 1.7B 模型上几百步就收敛，总训练时间 ~2-3 小时。
**真实推理任务**：Olmo 3 32B Think 训练了 21 天（224 GPUs）。

**差距来源**：
1. 任务复杂度：spell_backward vs 数学推理
2. 模型规模：1.7B vs 32B
3. 数据规模：3000 prompts vs 大规模推理数据

### 8.5 "Rejection Sampling 是免费的午餐" -- 有条件

**前提条件**：
1. 需要一个好的 RM（训练 RM 需要成本）
2. 需要生成多个 completions（推理成本 = N * 单次推理）
3. 只在 RM 评估可靠的分布内有效

**实际成本**：`num_completions_per_prompt: 8` 意味着推理成本是直接生成的 8 倍。

---

## 待更新

> 以下内容将在实际运行实验后补充：

- [ ] 实际显存占用数据（vs 预估）
- [ ] 实际训练时间数据
- [ ] 各算法在 spell_backward 上的收敛对比
- [ ] DPO beta sweep 的实际结果
- [ ] GRPO num_rollouts ablation 的实际结果
- [ ] Rejection Sampling 的 top vs random 对比
- [ ] 多 GPU 并行实验的实际效率

---

*本文档随复现进度持续更新。*
*配套文档：REPRODUCTION_PLAN.md（复现计划）、INTERVIEW_PREP.md（知识体系）、INTERVIEW_DEEP_DIVE.md（面试应答）*
