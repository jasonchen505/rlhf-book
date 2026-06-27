# RLHF Book 全流程复现计划

> 硬件：8x NVIDIA RTX 4090 (24GB VRAM each, 192GB total)
> 目标：完整复现项目所有 5 个代码模块 + 验证书中核心结论

---

## 一、硬件能力评估

### 1.1 单卡 4090 能力

| 指标 | 数值 |
|------|------|
| VRAM | 24GB |
| FP16/BF16 算力 | ~82.6 TFLOPS |
| 支持 Flash Attention | 是 (x86_64 + CUDA 12.x) |
| NVLink | 否（PCIe 4.0 x16，卡间通信带宽 ~32GB/s） |

### 1.2 项目模型 vs 显存匹配

| 模型 | 参数量 | BF16 显存 | 训练显存 (含 optimizer) | 单卡 4090 |
|------|--------|----------|----------------------|-----------|
| Qwen3-0.6B | 0.6B | ~1.2GB | ~4-6GB | 完全 OK |
| OLMo-2-1B | 1B | ~2GB | ~8-10GB | 完全 OK |
| Qwen3-1.7B | 1.7B | ~3.4GB | ~10-15GB | 完全 OK |
| Qwen2.5-3B | 3B | ~6GB | ~20-25GB | 勉强 OK (需 gradient checkpointing) |
| AceMath-7B-RM | 7B | ~14GB | ~20-24GB | 推理 OK，训练需谨慎 |

**结论**：项目默认所有模型都能在单卡 4090 上跑。8 张卡的核心价值是**并行实验**而非分布式训练。

### 1.3 关键限制

1. **无 NVLink**：4090 之间只有 PCIe，做 tensor parallel 效率很低
2. **单卡 24GB**：无法跑 7B+ 模型的 full finetune（需 LoRA 或量化）
3. **代码是单 GPU 设计**：没有内置 FSDP/DeepSpeed 支持

---

## 二、复现策略：并行实验 + 全流程覆盖

### 2.1 核心策略

```
GPU 0-1: Phase 1 - SFT 实验
GPU 2-3: Phase 2 - Reward Model 实验
GPU 4-5: Phase 3 - Policy Gradient 实验
GPU 6-7: Phase 4 - DPO / Rejection Sampling 实验
```

每个 Phase 完成后轮换，最终在所有 GPU 上做对比实验。

### 2.2 时间线总览

| 阶段 | 内容 | GPU | 预计时间 |
|------|------|-----|---------|
| Day 0 | 环境搭建 + 数据下载 | 全部 | 2-3 小时 |
| Day 1 | SFT 训练 + 超参 sweep | GPU 0-1 | 4-6 小时 |
| Day 2 | Reward Model 训练 (Preference/ORM/PRM) | GPU 2-3 | 4-6 小时 |
| Day 3 | Policy Gradients (REINFORCE/RLOO/GRPO/PPO) | GPU 4-7 | 6-8 小时 |
| Day 4 | DPO + 变体 (IPO/KTO/APO/SimPO) | GPU 0-3 | 4-6 小时 |
| Day 5 | Rejection Sampling + 全流程串联 | GPU 4-7 | 4-6 小时 |
| Day 6 | 对比实验 + 分析 + 文档 | 全部 | 4-6 小时 |

**总计：约 6 天，每天 4-8 小时**

---

## 三、Day 0：环境搭建

### 3.1 基础环境

```bash
# 1. 确认 CUDA 和 GPU 状态
nvidia-smi  # 确认 8 张 4090 都可见

# 2. 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 克隆项目
cd /data/home/yizhou
git clone https://github.com/natolambert/rlhf-book.git
cd rlhf-book/code

# 4. 安装依赖（含 flash-attn）
uv sync --extra flash --extra dev

# 5. 验证安装
uv run python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

### 3.2 数据预下载

```bash
# 预下载所有需要的数据集，避免训练时卡住
uv run python -c "
from datasets import load_dataset
# SFT 数据
load_dataset('HuggingFaceH4/no_robots', split='train')
# 偏好数据
load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
# ORM 数据
load_dataset('openai/gsm8k', 'main', split='train')
# PRM 数据 (需要 HF token)
# load_dataset('RLHF-Workshop/prm800k', split='train')
print('Done!')
"
```

### 3.3 模型预下载

```bash
# 预下载所有模型权重
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
models = [
    'allenai/OLMo-2-0425-1B',
    'allenai/OLMo-2-0425-1B-SFT',
    'Qwen/Qwen3-0.6B-Base',
    'Qwen/Qwen3-1.7B',
]
for m in models:
    print(f'Downloading {m}...')
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype='auto')
    print(f'  Done: {m}')
"
```

### 3.4 W&B 配置

```bash
# 设置 W&B（可选，设为 disabled 则本地日志）
export WANDB_PROJECT=rlhf-book
export WANDB_API_KEY=your_key_here
# 或者禁用：
# export WANDB_MODE=disabled
```

---

## 四、Day 1：指令微调 (SFT)

### 4.1 实验 1：标准 SFT（GPU 0）

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m instruction_tuning.train     --config instruction_tuning/configs/sft_olmo2_1b.yaml
```

**预期时间**：~1-2 小时（9.5K 样本，3 epochs）
**观察指标**：loss 曲线、in-loop sample panels（base model -> assistant transition）

### 4.2 实验 2：学习率 sweep（GPU 1，与实验 1 并行）

创建新配置文件 `sft_lr_sweep.yaml`：

```bash
# 复制配置并修改 LR
cp instruction_tuning/configs/sft_olmo2_1b.yaml /tmp/sft_lr_3e-5.yaml
# 修改 lr: 3.0e-5

CUDA_VISIBLE_DEVICES=1 uv run python -m instruction_tuning.train     --config /tmp/sft_lr_3e-5.yaml
```

### 4.3 实验 3-4：更多 sweep（GPU 0-1，实验 1-2 完成后）

| 实验 | GPU | 变量 | 目的 |
|------|-----|------|------|
| 3 | 0 | lr=1e-5, epochs=5 | 过拟合测试 |
| 4 | 1 | max_samples=2000 | 数据量影响 |

### 4.4 SFT 关键验证点

1. **base -> assistant transition**：step 0 模型乱续写，step 650 能正确回答并停止
2. **loss 曲线**：应该稳步下降，无 NaN
3. **in-loop samples**：每 50 步生成一次，观察格式变化

---

## 五、Day 2：奖励模型训练

### 5.1 实验 5：Preference RM（GPU 2）

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m reward_models.train_preference_rm     --samples 5000 --epochs 1
```

**预期时间**：~1-2 小时
**观察指标**：chosen-rejected margin、accuracy

### 5.2 实验 6：ORM（GPU 3，与实验 5 并行）

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m reward_models.train_orm     --samples 400 --epochs 2
```

**预期时间**：~30 分钟
**观察指标**：correct vs perturbed answer 的分数差异

### 5.3 实验 7：PRM（GPU 2，实验 5 完成后）

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m reward_models.train_prm     --samples 500 --epochs 2
```

### 5.4 实验 8：RM 超参实验（GPU 3，实验 6 完成后）

```bash
# 更多样本 + 更多 epoch
CUDA_VISIBLE_DEVICES=3 uv run python -m reward_models.train_preference_rm     --samples 10000 --epochs 2
```

### 5.5 RM 关键验证点

1. **margin 上升**：chosen-rejected reward 差值应该持续增大
2. **accuracy 稳定**：通常 0.65-0.75 为合理范围
3. **不过拟合**：只训练 1 epoch 是标准做法

---

## 六、Day 3：策略梯度方法（最核心）

### 6.1 实验 9：GRPO（GPU 4）—— 推荐首先跑

```bash
CUDA_VISIBLE_DEVICES=4 uv run python -m policy_gradients.train     --config policy_gradients/configs/grpo.yaml
```

**预期时间**：~2-3 小时
**关键指标**：`avg_correctness`, `avg_format`, `avg_binary`

### 6.2 实验 10：REINFORCE（GPU 5，与实验 9 并行）

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m policy_gradients.train     --config policy_gradients/configs/reinforce.yaml
```

### 6.3 实验 11：RLOO（GPU 6，与实验 9-10 并行）

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -m policy_gradients.train     --config policy_gradients/configs/rloo.yaml
```

### 6.4 实验 12：PPO（GPU 7，与实验 9-11 并行）

```bash
CUDA_VISIBLE_DEVICES=7 uv run python -m policy_gradients.train     --config policy_gradients/configs/ppo.yaml
```

### 6.5 实验 13-16：更多算法（GPU 4-7，前一批完成后）

| 实验 | GPU | 算法 | 特点 |
|------|-----|------|------|
| 13 | 4 | Dr. GRPO | 去 std 归一化 |
| 14 | 5 | GSPO | sequence-level IS ratio |
| 15 | 6 | CISPO | clip weights not objective |
| 16 | 7 | DAPO | decoupled clip + dynamic sampling |

### 6.6 实验 17-20：GRPO ablation（GPU 4-7）

| 实验 | 变量 | 目的 |
|------|------|------|
| 17 | num_rollouts=4 | 组大小影响 |
| 18 | num_rollouts=16 | 更大组 |
| 19 | temperature=0.3 vs 1.0 | 采样温度影响 |
| 20 | beta=0.1 vs 0.0 | KL 惩罚影响 |

### 6.7 Policy Gradient 关键验证点

1. **avg_correctness 上升**：模型学会 spell_backward
2. **group contrast**：同一个 prompt 下，部分对部分错（对比信号）
3. **算法对比**：GRPO vs REINFORCE 的收敛速度和稳定性
4. **KL 轨迹**：beta>0 时 KL 应该被控制在合理范围

---

## 七、Day 4：直接对齐算法

### 7.1 实验 21：DPO（GPU 0）

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m direct_alignment.train     --config direct_alignment/configs/dpo.yaml
```

**预期时间**：~1-2 小时
**关键指标**：accuracy, margins, chosen_rewards, rejected_rewards

### 7.2 实验 22-24：DPO 变体（GPU 1-3，并行）

```bash
# GPU 1: IPO
CUDA_VISIBLE_DEVICES=1 uv run python -m direct_alignment.train     --config direct_alignment/configs/ipo.yaml

# GPU 2: KTO
CUDA_VISIBLE_DEVICES=2 uv run python -m direct_alignment.train     --config direct_alignment/configs/kto.yaml

# GPU 3: APO-Zero
CUDA_VISIBLE_DEVICES=3 uv run python -m direct_alignment.train     --config direct_alignment/configs/apo_zero.yaml
```

### 7.3 实验 25-26：DPO beta sweep（GPU 0-1，前一批完成后）

| 实验 | GPU | beta | 目的 |
|------|-----|------|------|
| 25 | 0 | 0.01 | 低 beta：过优化风险 |
| 26 | 1 | 0.5 | 高 beta：保守更新 |

### 7.4 实验 27：快速 DPO 测试（GPU 2，前一批完成后）

```bash
CUDA_VISIBLE_DEVICES=2 uv run python -m direct_alignment.train     --loss dpo --max_samples 1000
```

### 7.5 DPO 关键验证点

1. **accuracy 上升但不到 1.0**：0.6-0.8 是健康范围，到 1.0 说明过拟合
2. **margins 变化**：chosen_rewards 应上升，rejected_rewards 应下降
3. **beta 影响**：低 beta 训练快但可能过优化，高 beta 保守但安全

---

## 八、Day 5：拒绝采样 + 全流程串联

### 8.1 实验 28：Rejection Sampling 预处理（GPU 4）

```bash
CUDA_VISIBLE_DEVICES=4 uv run python -m rejection_sampling.preprocess     --config rejection_sampling/configs/top_per_prompt.yaml
```

**预期时间**：~30 分钟（生成 + 打分）

### 8.2 实验 29-32：四种 RS 策略（GPU 4-7）

```bash
# GPU 4: top_per_prompt
CUDA_VISIBLE_DEVICES=4 uv run python -m rejection_sampling.train     --config rejection_sampling/configs/top_per_prompt.yaml

# GPU 5: random_per_prompt (baseline)
CUDA_VISIBLE_DEVICES=5 uv run python -m rejection_sampling.train     --config rejection_sampling/configs/random_per_prompt.yaml

# GPU 6: top_k_overall
CUDA_VISIBLE_DEVICES=6 uv run python -m rejection_sampling.train     --config rejection_sampling/configs/top_k_overall.yaml

# GPU 7: random_k_overall (baseline)
CUDA_VISIBLE_DEVICES=7 uv run python -m rejection_sampling.train     --config rejection_sampling/configs/random_k_overall.yaml
```

### 8.3 全流程串联验证

完整 RLHF 流水线：
```
SFT (Day 1) -> RM (Day 2) -> GRPO (Day 3) -> 模型
```

用 GRPO 训练好的模型做 Rejection Sampling：
1. 用 SFT 模型生成 completions
2. 用 Preference RM 打分
3. 选 top completions 做 SFT
4. 对比 GRPO 模型 vs RS 模型 vs 原始 SFT 模型

---

## 九、Day 6：对比分析 + 扩展实验

### 9.1 核心对比表

| 方法 | 类型 | 需要 RM | 在线生成 | 计算成本 | 适用场景 |
|------|------|---------|---------|---------|---------|
| SFT | 监督 | 否 | 否 | 低 | 基础格式 |
| DPO | 离线偏好 | 否 | 否 | 低 | 风格对齐 |
| GRPO | 在线 RL | 是 | 是 | 中 | 推理/代码 |
| PPO | 在线 RL | 是+Value | 是 | 高 | 通用 |
| RS | 推理时 | 是 | 是 | 中 | 质量提升 |

### 9.2 扩展实验（如果有时间）

**实验 33：更大模型**
```bash
# Qwen3-0.6B 的 GRPO
# 修改 grpo.yaml 中 model_name: Qwen/Qwen3-0.6B-Base
CUDA_VISIBLE_DEVICES=0 uv run python -m policy_gradients.train     --config /tmp/grpo_0.6b.yaml
```

**实验 34：不同任务**
```bash
# 修改 spell_backward 的 word_len 范围
# min_word_len: 5, max_word_len: 15
```

**实验 35：KL 惩罚 ablation**
```bash
# beta=0.0 (无 KL), beta=0.01, beta=0.1
# 观察生成文本质量变化
```

---

## 十、并行调度详细表

### 10.1 GPU 利用率优化

```
Day 1 AM: GPU[0]=SFT_lr5e6  GPU[1]=SFT_lr3e5
Day 1 PM: GPU[0]=SFT_lr1e5  GPU[1]=SFT_small_data
Day 2 AM: GPU[2]=PrefRM     GPU[3]=ORM
Day 2 PM: GPU[2]=PRM        GPU[3]=RM_sweep
Day 3 AM: GPU[4]=GRPO       GPU[5]=REINFORCE  GPU[6]=RLOO  GPU[7]=PPO
Day 3 PM: GPU[4]=Dr.GRPO    GPU[5]=GSPO       GPU[6]=CISPO GPU[7]=DAPO
Day 4 AM: GPU[0]=DPO        GPU[1]=IPO        GPU[2]=KTO   GPU[3]=APO
Day 4 PM: GPU[0]=DPO_b0.01  GPU[1]=DPO_b0.5   GPU[2]=DPO_1k
Day 5 AM: GPU[4]=RS_preprocess -> GPU[4-7]=RS_4_strategies
Day 5 PM: GPU[4-7]=GRPO_ablation (rollouts, temp, beta)
Day 6:    全部 = 对比分析 + 扩展实验
```

### 10.2 实验记录模板

每个实验记录：
```
实验编号: XX
GPU: X
配置文件: xxx.yaml
命令: uv run python -m xxx.train --config xxx
开始时间: 
结束时间:
最终指标:
- loss: X
- accuracy: X
- 其他关键指标: X
W&B 链接: xxx
观察/问题: xxx
```

---

## 十一、预期产出

### 11.1 完整复现清单

- [ ] SFT: OLMo-2-1B on No Robots (3 个 LR 对比)
- [ ] Preference RM: Bradley-Terry on UltraFeedback
- [ ] ORM: GSM8K binary classification
- [ ] PRM: PRM800K step-level classification
- [ ] REINFORCE: spell_backward
- [ ] RLOO: spell_backward
- [ ] GRPO: spell_backward + ablation
- [ ] PPO: spell_backward
- [ ] Dr. GRPO, GSPO, CISPO, DAPO: spell_backward
- [ ] DPO: UltraFeedback
- [ ] IPO, KTO, APO: UltraFeedback
- [ ] DPO beta sweep
- [ ] Rejection Sampling: 4 strategies on GSM8K
- [ ] 全流程串联: SFT -> RM -> GRPO -> RS

### 11.2 学习产出

- 理解每个算法的数学原理和实现细节
- 掌握超参对训练的影响（LR, beta, num_rollouts, temperature）
- 积累工程经验（显存管理、并行实验、debugging）
- 形成对 RLHF 各方法 trade-off 的直觉

---

## 十二、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| OOM (模型太大) | 低 | 单实验失败 | 降低 batch_size，开 gradient checkpointing |
| 训练 NaN | 中 | 单实验失败 | 降低 LR，检查数据 |
| GPU 故障 | 低 | 减少并行度 | 用剩余 GPU 继续 |
| 数据下载慢 | 中 | 延期 | Day 0 预下载 |
| W&B 不可用 | 低 | 无影响 | 设 WANDB_MODE=disabled |

---

*基于 RLHF Book 项目代码分析和 8x 4090 硬件能力制定。*
*配套文档：INTERVIEW_PREP.md（知识体系）、INTERVIEW_DEEP_DIVE.md（面试应答）、REPRODUCTION_LEARNING.md（增量学习笔记）*
