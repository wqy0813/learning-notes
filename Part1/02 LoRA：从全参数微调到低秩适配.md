---
title: LoRA：从全参数微调到低秩适配
aliases:
  - LoRA
  - Low-Rank Adaptation
tags:
  - 深度学习
  - 大语言模型
  - 参数高效微调
  - SFT
---

# LoRA：从全参数微调到低秩适配

> [!abstract] 一句话理解
> **LoRA 不修改原来的大模型权重，而是在旁边增加一个很小的“修正通道”，只训练这个修正通道，让模型适应新任务。**

## 1. LoRA 为什么出现？

预训练大模型已经学到了大量通用知识。

如果针对一个新任务，例如医疗领域 SFT，直接进行**全参数微调（Full Fine-Tuning）**，就需要更新模型中大量原有参数，训练时还需要保存这些参数对应的梯度和优化器状态，成本很高。

全参数微调可以写成：
$$
W' = W + \Delta W
$$
其中：

- $W$：预训练得到的原始权重；
- $\Delta W$：为了适应新任务而产生的权重变化；
- $W'$：微调后的权重。

全参数微调直接更新 $W$。

LoRA 的想法是：

> **冻结原来的 $W$，不直接训练它，只学习一个较小的 $\Delta W$。**

---

## 2. 先统一矩阵约定：统一使用 $BA$

为了和 LoRA 原论文及大多数资料保持一致，本文统一使用**列向量**约定：
$$
x\in\mathbb{R}^{D_{in}\times1}
$$
$$
W\in\mathbb{R}^{D_{out}\times D_{in}}
$$
原始线性层：
$$
y=Wx
$$
在这个约定下，LoRA 的低秩更新统一写成：
$$
\boxed{\Delta W=BA}
$$
其中：
$$
B\in\mathbb{R}^{D_{out}\times r}
$$
$$
A\in\mathbb{R}^{r\times D_{in}}
$$
于是：
$$
\boxed{W'=W+BA}
$$
对应的前向计算为：
$$
\boxed{y=Wx+BAx}
$$
注意：

> **LoRA 不是把原来的 $W$ 拆成 $A$ 和 $B$。**
>
> 原来的 $W$ 保持冻结，$BA$ 专门表示为了新任务需要学习的权重变化。

### LoRA 的核心结构

<div align="center">
<svg width="720" height="310" viewBox="0 0 720 310" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="lora-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <rect x="302" y="15" width="116" height="38" rx="8" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="360" y="39" text-anchor="middle" font-size="14" fill="#334155">输入 x</text>

  <path d="M360 53 V75 H205 V96" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>
  <path d="M360 75 H515 V96" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>

  <rect x="105" y="96" width="200" height="62" rx="9" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="205" y="121" text-anchor="middle" font-size="14" font-weight="600" fill="#334155">原始权重 W</text>
  <text x="205" y="143" text-anchor="middle" font-size="12" fill="#64748b">冻结 · 不更新</text>

  <rect x="415" y="96" width="200" height="42" rx="9" fill="#eef8f7" stroke="#5f9ea0"/>
  <text x="515" y="122" text-anchor="middle" font-size="14" font-weight="600" fill="#334155">A · 可训练</text>

  <path d="M515 138 V157" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>

  <rect x="415" y="157" width="200" height="42" rx="9" fill="#eef8f7" stroke="#5f9ea0"/>
  <text x="515" y="183" text-anchor="middle" font-size="14" font-weight="600" fill="#334155">B · 可训练</text>

  <text x="515" y="220" text-anchor="middle" font-size="12" fill="#64748b">低秩路径：Dᵢₙ → r → Dₒᵤₜ</text>

  <path d="M205 158 V248 H341" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>
  <path d="M515 199 V248 H379" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>

  <circle cx="360" cy="248" r="19" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="360" y="255" text-anchor="middle" font-size="22" fill="#334155">+</text>

  <path d="M360 267 V282" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#lora-arrow)"/>
  <text x="360" y="302" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">最终输出 y = Wx + BAx</text>
</svg>
</div>

---

## 3. 为什么叫 Low-Rank？

LoRA 使用两个较小的矩阵：
$$
B\in\mathbb{R}^{D_{out}\times r}
$$
$$
A\in\mathbb{R}^{r\times D_{in}}
$$
因此：
$$
BA\in\mathbb{R}^{D_{out}\times D_{in}}
$$
虽然 $BA$ 最终和原始权重 $W$ 一样大，但它中间只有 $r$ 维。

> **人话：rank 可以近似理解为一个矩阵能够表达的独立方向的数量。LoRA 将权重变化 $\Delta W$ 表示成 $BA$，而 $BA$ 的计算过程是 $D_{in}\rightarrow r\rightarrow D_{out}$，中间只有 $r$ 个维度，所以它最多只能表达 $r$ 个独立方向。因此 $\Delta W$ 的秩需要满足 $\operatorname{rank}(\Delta W)=\operatorname{rank}(BA)\le r$。这就是 LoRA 为什么叫 Low-Rank（低秩）Adaptation。**

所以 LoRA 的核心就是：
$$
\boxed{\Delta W=BA}
$$
并且：
$$
\boxed{\operatorname{rank}(\Delta W)\le r}
$$
LoRA 的核心假设是：

> **为了让一个已经预训练好的模型适应某个下游任务，真正需要发生的权重变化 $\Delta W$，可能只需要较低的秩就能有效表示。**

## 4. 为什么两个小矩阵能减少参数？

如果直接训练完整的 $\Delta W$：
$$
D_{in}\times D_{out}
$$
需要：
$$
D_{in}D_{out}
$$
个参数。

LoRA 只训练 $A$ 和 $B$：
$$
D_{in}r+rD_{out}
$$
即：
$$
\boxed{
N_{LoRA}=r(D_{in}+D_{out})
}
$$
例如：
$$
D_{in}=D_{out}=4096,\qquad r=8
$$
完整矩阵：
$$
4096\times4096=16{,}777{,}216
$$
LoRA：
$$
4096\times8+8\times4096=65{,}536
$$
所以单个线性层上，LoRA 的可训练参数远少于完整权重。

这里比较的是**参数数量**，不等于整次训练的峰值显存。

---

## 5. rank $r$ 到底是什么？

可以把 $r$ 理解成：

> **LoRA 用多大的低维空间去学习“模型需要发生的变化”。**

例如：
$$
4096\rightarrow r\rightarrow4096
$$
当：
$$
r=8
$$
就是：
$$
4096\rightarrow8\rightarrow4096
$$
所以：

- $r$ 较小：参数少，更新空间受到更强限制；
- $r$ 较大：参数更多，能够表达更复杂的更新；
- $r$ 增大不代表效果一定更好，只代表 LoRA 的更新容量更大。

在目标层不变时：
$$
N_{LoRA}\propto r
$$
---

## 6. LoRA 放到 Transformer 哪里？

你已经学过 Attention 中的线性投影：
$$
q=W_Qx
$$
$$
k=W_Kx
$$
$$
v=W_Vx
$$
LoRA 可以添加到这些线性层，也可以添加到 MLP 中的线性层。

例如对 $W_Q$ 添加 LoRA：
$$
q=(W_Q+\Delta W_Q)x
$$
令：
$$
\Delta W_Q=B_QA_Q
$$
得到：
$$
q=W_Qx+B_QA_Qx
$$
也就是：
$$
\boxed{
q=q_{base}+q_{LoRA}
}
$$
然后继续正常进行 Attention：
$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$
所以：

> **LoRA 不改变 Transformer 的整体结构，也不改变 Attention 的核心公式。**
>
> 它只是改变某些线性层的权重计算。

---

## 7. LoRA 的完整前向过程

加入 LoRA 的线性层可以写成：
$$
\boxed{
y=Wx+\frac{\alpha}{r}BAx
}
$$
其中：

- $W$：冻结的原始权重；
- $A,B$：可训练的 LoRA 参数；
- $r$：rank；
- $\alpha$：LoRA scaling 参数。

可以看成两条路径：

原始模型：
$$
x\rightarrow W\rightarrow Wx
$$
LoRA：
$$
x\rightarrow A\rightarrow B\rightarrow BAx
$$
最后：
$$
\boxed{
y=y_{base}+\frac{\alpha}{r}y_{LoRA}
}
$$
注意：这里统一使用 $BA$，这样后面看到论文、教程和代码时，不需要在 $AB/BA$ 之间来回转换。

---

## 8. LoRA 怎么训练？

训练主线和普通神经网络一样：
$$
\text{输入}
\rightarrow
\text{前向传播}
\rightarrow
\text{Loss}
\rightarrow
\text{反向传播}
\rightarrow
\text{更新参数}
$$
区别只有：
$$
W:\ \text{冻结}
$$
$$
A,B:\ \text{训练}
$$
因此：
$$
Loss
\rightarrow
\text{反向传播}
\rightarrow
A,B
$$
优化器只更新 LoRA 参数。

### 一个重要概念

**冻结 $W$ ≠ 反向传播被切断。**

冻结的意思是：

> $W$ 的数值不被优化器更新。

为了计算 Loss 对 $A$、$B$ 的梯度，自动微分仍然需要经过整个相关计算路径。

---

## 9. 为什么 $B$ 通常初始化为 0？

LoRA 的经典初始化：
$$
A\sim\text{随机初始化}
$$
$$
B=0
$$
于是训练开始时：
$$
BA=0
$$
所以：
$$
y=Wx+\frac{\alpha}{r}BAx
$$
变成：
$$
\boxed{y=Wx}
$$
也就是说：

> **刚加入 LoRA 时，模型的输出与原始模型一致。**

然后训练开始：
- 第一次反向传播时，$B$ 通常能够获得非零梯度；
- 因为此时 $B=0$，$A$ 的梯度通常为 0；
- $B$ 更新后，后续训练中 $A$ 也能够获得梯度。

因此不能简单地把：
$$
A=0,\qquad B=0
$$
否则 LoRA 分支一开始没有有效的学习信号。

补充：LoRA 原论文采用随机高斯初始化 $A$、零初始化 $B$；当前 Hugging Face PEFT 默认也是 $B=0$，但 $A$ 的默认初始化方式是 Kaiming-uniform。因此学习原理时记住“$A$ 随机、$B=0$”即可，实际实现时以具体框架配置为准。

---

## 10. alpha 是什么？

LoRA 的前向公式：
$$
y=Wx+\frac{\alpha}{r}BAx
$$
其中：
- $r$控制：**LoRA 更新的容量**
- $\alpha$控制：**LoRA 分支的缩放程度**

所以经典 LoRA 中：
$$
\frac{\alpha}{r}
$$
就是 LoRA 分支的 scaling。

注意：
$$
\boxed{\alpha\neq learning\ rate}
$$
alpha 不是学习率。

---

## 11. LoRA 和 SFT 是什么关系？

这两个概念解决的是不同的问题。

**SFT：**

> 用什么数据、什么训练目标训练模型？

例如医疗 SFT：
$$
\text{医疗问题}
\rightarrow
\text{标准医疗回答}
$$
**LoRA：**

> 训练模型时，具体更新哪些参数？

所以可以组合：
$$
\boxed{\text{医疗 SFT}+\text{LoRA}}
$$
意思就是：

> **使用医疗数据做 SFT，但只训练 LoRA Adapter，不更新整个大模型。**

---

## 12. 训练完成后怎么办？

训练完成后得到：
$$
A^*,\qquad B^*
$$
基座模型的权重仍然是：
$$
W
$$
LoRA 学到的权重变化为：
$$
\boxed{
\Delta W=\frac{\alpha}{r}BA
}
$$
### 方式一：Base Model + LoRA Adapter

推理时使用：
$$
W+\frac{\alpha}{r}BA
$$
可以在同一个基座模型上加载不同任务的 Adapter。

例如：
$$
\text{Base Model}
+
\begin{cases}
\text{医疗 LoRA}\\
\text{法律 LoRA}\\
\text{代码 LoRA}
\end{cases}
$$
这样可以只保存每个任务对应的小型 Adapter。

### 方式二：合并权重

也可以直接计算：
$$
\boxed{
W_{merged}
=
W+\frac{\alpha}{r}BA
}
$$
然后使用合并后的权重进行普通前向计算。

LoRA 原论文的一个重要工程优势就是可以将 $BA$ 合并回冻结权重，从而不增加额外的推理层级。

---

## 13. LoRA 的完整主线

把整件事情压缩成一条线：
$$
\begin{gathered}
\text{预训练模型}\\[4pt]
\downarrow\\[4pt]
\text{冻结原始权重 }W\\[4pt]
\downarrow\\[4pt]
\text{加入低秩矩阵 }A,B\\[4pt]
\downarrow\\[4pt]
\Delta W=BA\\[4pt]
\downarrow\\[4pt]
y=Wx+\frac{\alpha}{r}BAx\\[4pt]
\downarrow\\[4pt]
\text{前向传播}\\rightarrow\text{Loss}\\[4pt]
\downarrow\\[4pt]
\text{反向传播}\\[4pt]
\downarrow\\[4pt]
\text{只更新 }A,B\\[4pt]
\downarrow\\[4pt]
W_{\mathrm{adapted}}=W+\frac{\alpha}{r}BA
\end{gathered}
$$
一句话：

> **LoRA 就是在不动原模型的情况下，增加一个很小的低秩修正通道，让这个通道学习新任务需要的变化。**

---

## 14. 这一阶段真正需要记住的公式

### ① LoRA 核心
$$
\boxed{\Delta W=BA}
$$
### ② LoRA 前向
$$
\boxed{
y=Wx+\frac{\alpha}{r}BAx
}
$$
### ③ 参数量
$$
\boxed{
N_{LoRA}
=
r(D_{in}+D_{out})
}
$$
### ④ 秩
$$
\boxed{
\operatorname{rank}(\Delta W)\le r
}
$$
### ⑤ 合并
$$
\boxed{
W_{merged}
=
W+\frac{\alpha}{r}BA
}
$$
### ⑥ 常见初始化
$$
\boxed{
A\sim\text{随机初始化},
\qquad
B=0
}
$$
---

## 15. 从列向量公式到 PyTorch 代码

前面统一使用**列向量约定**：

$$
\boxed{y=Wx+\frac{\alpha}{r}BAx}
$$

PyTorch 的 `nn.Linear` 通常把一个 Batch 中的样本按**行**排列。设 Batch Size 为 $M$，则：

$$
X\in\mathbb R^{M\times D_{in}}
$$

对应的 Batch 前向公式为：

$$
\boxed{Y=XW^T+\frac{\alpha}{r}(XA^T)B^T}
$$

这只是输入的排列方式不同，没有改变第 2 节的 $BA$ 约定。对应关系如下：

| 列向量公式 | PyTorch（Batch） |
| --- | --- |
| $Wx$ | `x @ W.T` |
| $Ax$ | `x @ A.T` |
| $B(Ax)$ | `(x @ A.T) @ B.T` |
| $Wx+\frac{\alpha}{r}BAx$ | `base_out + scaling * lora_out` |

其中 $B$ 为 Batch Size、$D_{in}$ 为输入特征维度、$D_{out}$ 为输出特征维度、$r$ 为 LoRA 秩。对应的 Shape 主线是：

```text
X [M, D_in]
├─ 原始分支：X @ W.T   [B, D_in] @ [D_in, D_out] → [B, D_out]   （W 冻结）
└─ LoRA 分支：X @ A.T  [B, D_in] @ [D_in, r]     → [B, r]
                ↓ @ B.T           [B, r] @ [r, D_out] → [B, D_out]
                ↓ × α/r
               两条分支相加 → Y [M, D_out]
```

### 15.1 手写一个最简 LoRALinear

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()

        # 教学示例：实际微调应传入已有预训练权重的线性层
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)  # 冻结 W

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.base(x)                 # XW^T
        lora_out = (x @ self.A.T) @ self.B.T   # (XA^T)B^T
        return base_out + self.scaling * lora_out
```

| PyTorch API | 对应作用 |
| --- | --- |
| `nn.Linear(in_features, out_features)` | 创建原始线性层；权重 Shape 是 $[D_{out},D_{in}]$ |
| `requires_grad_(False)` | 冻结 W，不计算 W 的参数梯度 |
| `nn.Parameter(...)` | 将 A、B 注册为可训练参数 |
| `forward(x)` | 原始分支与低秩分支相加，输出 Shape 不变 |

这里新创建的 `self.base` **没有预训练知识**，仅用于理解结构。实际 LoRA 微调应将其替换为已经加载预训练权重的目标线性层。

初始化 $A$ 为随机值、$B$ 为零，因此起始时 $BA=0$，LoRA 不改变原始输出。

---

## 16. PyTorch 如何训练 LoRA 参数

训练流程没有改变：

$$
\text{前向传播}\ \rightarrow\ \text{Loss}\ \rightarrow\
\text{反向传播}\ \rightarrow\ \text{优化器更新}
$$

下面用 MSE Loss 演示一个**教学用线性层**的训练；真正的语言模型 SFT 通常使用 Token 级 CrossEntropy Loss。

```python
torch.manual_seed(0)
model = LoRALinear(4, 3, r=2, alpha=2)

W_before = model.base.weight.detach().clone()
optimizer = torch.optim.AdamW([model.A, model.B], lr=1e-3)

x = torch.randn(5, 4)
target = torch.randn(5, 3)

for step in range(2):
    optimizer.zero_grad()                       # 清除旧梯度
    prediction = model(x)                      # 前向
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()                            # 计算 A、B 的梯度

    print(step, model.A.grad.norm().item(),
          model.B.grad.norm().item())
    optimizer.step()                           # 只更新 A、B

print(torch.equal(W_before, model.base.weight))  # 预期 True
```

- `optimizer.zero_grad()`：清空此前累积的梯度。
- `loss.backward()`：计算梯度并写入参数的 `.grad`，**此时不更新参数**。
- `optimizer.step()`：按梯度更新交给优化器的 A、B。
- 冻结 W 不会切断通向更早可训练层的反向传播。

特别注意第 9 节的零初始化：第一次反向传播时，$B=0$ 使 $\partial L/\partial A=0$；但 B 通常可以获得非零梯度。B 首次更新后，A 在后续步骤中通常也能获得梯度。

---

## 17. PEFT 如何自动添加 LoRA

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 的参数高效微调工具库。它根据配置，在选中的 Transformer 线性层中添加 LoRA，无须手写替换每一层。

例如只对 Attention 的 Q、V 投影添加 LoRA（仍按本文列向量约定）：

$$
q=W_Qx+\frac{\alpha}{r}B_QA_Qx
$$

$$
v=W_Vx+\frac{\alpha}{r}B_VA_Vx
$$

同一个 Block 的 Q、V 使用不同的 A、B；不同 Block 中的 LoRA 参数也相互独立。K、O 以及 FFN 的线性层也可以按需要设置为目标模块。LoRA 不改变原有 Attention 的计算公式。

### 17.1 加载基础模型 → 配置 → 注入

以下使用本项目对应的 Qwen3-0.6B-Base 演示：

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 1. 加载已有的预训练模型（需要相关依赖、网络和足够内存）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base"
)

# 2. 指定 LoRA 参数与目标模块
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. 在匹配的线性层中自动添加 LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

`target_modules=["q_proj", "v_proj"]` 的意思是：**查找模型中名称匹配这两个后缀的模块**，分别添加 A、B；并不是全模型只添加一套 A、B。其他模型的层名称可能不同，应先检查模型结构。

> PEFT 文档：[配置模型和 Adapter](https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config)、[LoRA 配置](https://huggingface.co/docs/peft/main/package_reference/lora)。

### 17.2 理解 LoraConfig 的关键参数

| 参数 | 作用 |
| --- | --- |
| `r=8` | 低秩维度，决定每层 LoRA 更新的表达容量和参数数量 |
| `lora_alpha=16` | 经典 LoRA 缩放系数是 $\alpha/r=2$ |
| `target_modules` | 指定哪些投影层添加 LoRA |
| `lora_dropout=0.05` | 训练期间对 **LoRA 分支输入**使用 Dropout；不作用于原始 W 分支 |
| `bias="none"` | 不额外训练 bias |
| `task_type="CAUSAL_LM"` | 指定因果语言模型任务 |

采用 Dropout 后，单个目标线性层在**训练时**可表示为：

$$
y=Wx+\frac{\alpha}{r}BA\,\mathrm{Dropout}(x)
$$

评估和推理时 Dropout 关闭。具体配置与默认初始化参见 [PEFT LoRA 官方文档](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)。

### 17.3 如何检查 PEFT 添加了什么

```python
model.print_trainable_parameters()

for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        print("模块:", name)
        print("W:", module.base_layer.weight.shape)
        print("A:", module.lora_A["default"].weight.shape)
        print("B:", module.lora_B["default"].weight.shape)
        print("W 可训练:", module.base_layer.weight.requires_grad)
        break
```

默认 Adapter 名称一般是 `"default"`。对一个被适配的线性层，A、B 的可训练参数量依然是：

$$
\boxed{N_{LoRA}=r(D_{in}+D_{out})}
$$

`print_trainable_parameters()` 用于检查全模型可训练参数总量与占比，不能替代实际训练验证。

---

## 18. PEFT 的训练、保存与加载

PEFT 已经配置了哪些参数参与训练，因此优化器只需要选取 `requires_grad=True` 的参数。

下面是**单个训练 step 的代码片段**。`batch` 应由 SFT 数据预处理和 DataLoader 提供，至少需要匹配模型的 `input_ids` 与 `labels`；本节尚未构造数据集，因此不是独立可执行的完整训练脚本。

```python
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=2e-4
)

optimizer.zero_grad()
outputs = model(**batch)    # batch 已含 input_ids、labels 等
loss = outputs.loss
loss.backward()
optimizer.step()
```

Loss 仍由整个语言模型的预测得到。梯度通过整个计算图传播，但**优化器只更新选中的 LoRA A、B**（在当前 `bias="none"`、未指定其他可训练模块的配置下）。

### 保存 Adapter

```python
model.save_pretrained("./medical_lora_adapter")
```

典型输出文件：

```text
medical_lora_adapter/
├── adapter_config.json
├── adapter_model.safetensors
└── README.md                 # 可能同时生成模型卡
```

- `adapter_config.json`：保存 LoRA 配置和基座模型信息。
- `adapter_model.safetensors`：保存 Adapter 权重，通常**不包含冻结的基础模型 W**。

推理时要重新加载匹配的基座模型，再加载 Adapter：

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base"
)
inference_model = PeftModel.from_pretrained(
    base, "./medical_lora_adapter"
)
inference_model.eval()
```

参考：[PEFT checkpoint 格式](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)。

---

## 19. 与下一阶段 SFT 的衔接

这份笔记已经覆盖：**LoRA 数学原理 → PyTorch 线性层 → 冻结与训练 → PEFT 注入 → Adapter 保存与加载**。

下一阶段再单独学习医疗 SFT 的数据链路：

```text
医疗问答样本
    ↓ Chat Template + Tokenizer
input_ids / attention_mask / labels
    ↓ Transformer（指定层已添加 LoRA）
logits
    ↓ 对有效的目标 Token 计算 CrossEntropy
loss
    ↓ backward + optimizer.step
更新 LoRA 参数 A、B
```

`labels` 如何设置、Prompt 是否参与 Loss、Padding 与回答部分如何 Mask，以及如何构建 `batch`，属于下一节的 SFT 数据预处理与 Loss Mask。

QLoRA、4-bit NF4、rsLoRA、DoRA、PiSSA 等属于进一步的量化与 LoRA 变体，暂不混入这条工程实现主线。

## 来源

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) —— LoRA 原论文，低秩更新和冻结基础权重。
- [PyTorch：nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) —— Batch 输入布局及权重 Shape。
- [Hugging Face PEFT：LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) —— `LoraConfig` 的参数。
- [Hugging Face PEFT：LoRA 概念](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) —— 初始化、Scaling、Dropout 和变体。
- [Hugging Face PEFT：配置与模型](https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config) —— `get_peft_model()`、训练及加载。
- [Hugging Face PEFT：Checkpoint 格式](https://huggingface.co/docs/peft/main/developer_guides/checkpoint) —— Adapter 文件。
- [Qwen3-0.6B-Base 模型页](https://huggingface.co/Qwen/Qwen3-0.6B-Base) —— 示例预训练模型。
- [Hugging Face Transformers：Parameter-efficient fine-tuning](https://huggingface.co/docs/transformers/peft) —— Transformers 与 PEFT 结合使用。

资料核对日期：2026-10-19。
