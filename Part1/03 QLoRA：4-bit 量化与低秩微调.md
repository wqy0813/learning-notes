---
title: "QLoRA：4-bit 量化与低秩微调"
aliases:
  - QLoRA
  - Quantized LoRA
tags:
  - 深度学习
  - 大语言模型
  - 参数高效微调
  - 量化
  - SFT
---

# QLoRA：4-bit 量化与低秩微调

> [!abstract] 一句话理解
> **人话**：QLoRA和LoRA的区别只是QLoRA通过某种方式，减少了原始权重对显存的占用
> **QLoRA = LoRA + 4-bit 基础权重量化。** LoRA 冻结原始权重 $W$、只训练低秩矩阵 $A、B$；QLoRA 进一步压缩冻结的基础权重，以减少显存占用，训练逻辑基本不变。

前置知识：[[02 LoRA：从全参数微调到低秩适配]]

论文：[QLoRA: Efficient Finetuning of Quantized LLMs（NeurIPS 2023）](https://arxiv.org/abs/2305.14314)  
官方文档：[Hugging Face PEFT — Quantization](https://huggingface.co/docs/peft/developer_guides/quantization)

## 1. 为什么 LoRA 还需要 QLoRA？

LoRA 的前向公式为：

$$
\boxed{y=Wx+\frac{\alpha}{r}BAx}
$$

其中 $W$ 冻结，只训练 $A、B$。但是**冻结参数不等于不需要存储参数**：前向计算仍要使用 $W$ 完成 $Wx$。

QLoRA 将基础权重以 4-bit 形式量化存储。以 **7B（约 70 亿参数）且全部权重按相同精度存储**为理想化估算：

| 权重存储方式 | 每个权重 | 基础权重理论存储量 |
| --- | ---: | ---: |
| FP16 | 16 bit = 2 Bytes | 约 14 GB |
| 4-bit | 4 bit = 0.5 Bytes | 约 3.5 GB |

由 $16/4=4$ 可知，**4-bit 权重编码的理论存储成本是 FP16 的四分之一**。实际显存还包含缩放系数等量化元数据，以及激活值、LoRA 参数、梯度、优化器状态和计算缓冲区，不能直接把 3.5 GB 当成训练峰值显存。

## 2. 量化与反量化是什么？

**量化（Quantization）**：将高精度权重映射到有限数量的代表值，并存储对应的低 bit 编码。

4-bit 有 $2^4=16$ 种编码。这**不表示只能处理 16 个原始小数**，而是用编码、代表值和缩放信息来近似恢复权重。

### 2.1 最小例子：一个权重

假设原始权重 $w=0.73$，某个**假设的** 4-bit 量化表中，最接近它的代表值是 $0.72$，编码为 `1010`：

```text
原始权重 0.73
     │ 量化：找最接近的代表值
     ▼
4-bit 编码 1010  （存储）
     │ 反量化：查表
     ▼
近似权重 0.72    （用于计算）
```

量化误差为：

$$
|0.73-0.72|=0.01
$$

这里的 `1010 ↔ 0.72` **只为演示机制，不是真实 NF4 编码表**。反量化得到的是近似值，不保证完全恢复原始 $W$。

### 2.2 为什么还需要 Scale（缩放系数）？

实际 QLoRA 采用**分块量化**，为权重块保存缩放信息，使同一套有限编码能够表示不同范围的权重。

假设 $w=1.46$，当前块的 Scale 是 $s=2$：

1. 缩放：$w/s=1.46/2=0.73$；
2. 假设量化代表值为 $0.72$，存储其 4-bit 编码；
3. 反量化时乘回 Scale：$\hat w=0.72\times2=1.44$。

$$
\boxed{1.46\ \xrightarrow{\text{量化}}\ \text{4-bit 编码}\ \xrightarrow{\text{反量化}}\ 1.44}
$$

因此，**量化结果不只有 4-bit 编码，还包含恢复权重所需的量化辅助信息**（例如每块的 Scale）。

## 3. QLoRA 完整前向：比 LoRA 多了哪一步？

继续使用前一篇笔记的**列向量约定**：

$$
x\in\mathbb{R}^{D_{in}\times1},\quad
W\in\mathbb{R}^{D_{out}\times D_{in}}
$$

$$
A\in\mathbb{R}^{r\times D_{in}},\quad
B\in\mathbb{R}^{D_{out}\times r}
$$

先量化冻结的基础权重：

$$
\boxed{W_q=Q_{\mathrm{NF4}}(W)}
$$

- $Q_{\mathrm{NF4}}$：使用 NF4 方案执行 4-bit 量化；
- $W_q$：量化权重的编码与辅助信息，而不是一个普通的浮点矩阵。

计算时按需反量化：

$$
\boxed{\hat W=\operatorname{Dequant}(W_q)}
$$

- $\hat W$：近似恢复的浮点权重，逻辑 Shape 仍为 $[D_{out},D_{in}]$，但数值不一定与原始 $W$ 完全相同。

**最终前向公式：**

$$
\boxed{y=\hat Wx+\frac{\alpha}{r}BAx}
$$

### 前向 Shape 主线

```text
输入 x                                   [D_in, 1]
  ├─ 基础分支（冻结）
  │    W_q：4-bit NF4 存储                逻辑形状 [D_out, D_in]
  │      ↓ 按需反量化
  │    Ŵ：浮点计算表示（如 BF16）        [D_out, D_in]
  │      ↓ Ŵx
  │    基础分支输出                       [D_out, 1]
  │
  └─ LoRA 分支（可训练）
       A @ x                            [r, 1]
         ↓ B @ (A @ x)
       B @ A @ x                        [D_out, 1]
         ↓ × α/r
       LoRA 分支输出                    [D_out, 1]

两条分支相加 → y                        [D_out, 1]
```

对比：

$$
\begin{aligned}
\text{LoRA:}\quad &y=Wx+\frac{\alpha}{r}BAx\\[6pt]
\text{QLoRA:}\quad &y=\hat Wx+\frac{\alpha}{r}BAx
\end{aligned}
$$

> [!important] 存储精度 ≠ 计算精度
> QLoRA 通常以 **4-bit 存储**基础权重，矩阵乘法时使用 **BF16 等浮点计算精度**。底层算子可以按需反量化，不要求在显存中长期保存完整的 BF16 权重副本。QLoRA 因此不一定比普通 LoRA 计算得更快。

## 4. QLoRA 如何反向传播、更新参数？

1. 加载预训练模型，将可量化的基础线性层权重量化为 4-bit，并冻结。
2. 为选定线性层添加 LoRA 矩阵 $A、B$。
3. 前向计算 $y=\hat Wx+(\alpha/r)BAx$，得到 Loss。
4. 反向传播：梯度可以穿过冻结的量化基础模型传播到前面的可训练模块。
5. **只更新 LoRA 矩阵 $A、B$**，不更新基础权重 $W_q$。

用梯度下降示意：

$$
A\leftarrow A-\eta\nabla_A\mathcal L
$$

$$
B\leftarrow B-\eta\nabla_B\mathcal L
$$

其中 $\eta$ 是学习率，$\mathcal L$ 是训练损失；实际训练通常由 AdamW 等优化器更新适配器参数。

| 参数 | 是否更新 |
| --- | --- |
| 量化后的基础权重 $W_q$ | **不更新（冻结）** |
| LoRA 降维矩阵 $A$ | 更新 |
| LoRA 升维矩阵 $B$ | 更新 |

**QLoRA 不需要直接对离散的 4-bit 编码做梯度更新**。这也是它能够沿用 LoRA 训练方式的关键。

## 5. NF4、双重量化、分页优化器

| 技术 | 作用 | 需要记住的关键点 |
| --- | --- | --- |
| **NF4（NormalFloat 4-bit）** | 为近似正态分布的权重设计非均匀的 16 个量化值 | 降低此类权重分布的量化误差 |
| **Double Quantization（双重量化）** | 进一步量化 Scale 等量化常数 | 继续减少量化辅助信息的存储成本 |
| **Paged Optimizers（分页优化器）** | 利用统一内存管理优化器状态 | 缓解训练时的显存峰值 |

这三项见 QLoRA 原始论文 [1]。其中 NF4 决定了基础权重的量化方式，另外两项主要用于进一步控制内存开销。

## 6. 最终对比

| 对比 | LoRA | QLoRA |
| --- | --- | --- |
| 基础权重的典型存储 | FP16/BF16 | 4-bit NF4 |
| 基础权重是否冻结 | 是 | 是 |
| 可训练参数 | LoRA 的 $A、B$ | LoRA 的 $A、B$ |
| 基础分支的前向 | $Wx$ | $\hat Wx$（按需反量化） |
| 核心目的 | 降低可训练参数量 | 在 LoRA 基础上降低基础模型显存占用 |
| 额外代价 | 无 4-bit 反量化步骤 | 有量化误差和相关计算开销 |

**结论：** QLoRA 保留 LoRA 的低秩适配与参数更新逻辑，主要改动是将**冻结的基础权重压缩存储**，在前向计算时按需反量化。

## 来源

1. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers 等，NeurIPS 2023，访问于 2026-10-19。
2. [PEFT：Quantization](https://huggingface.co/docs/peft/developer_guides/quantization) — Hugging Face 官方文档，访问于 2026-10-19。
3. [Transformers：Bitsandbytes / QLoRA](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes) — Hugging Face 官方文档（main），访问于 2026-10-19。

资料核对日期：2026-10-19。

