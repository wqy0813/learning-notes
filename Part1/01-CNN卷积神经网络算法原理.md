## 一、神经元工作流程

结构单元：**神经元**

表达式：$a = h(w \cdot x + b)$

**工作流程：**

1. **对输入信号进行线性组合**，$\sum wx$（权重 $w$ 代表了该特征的重要性）
2. **引入偏置 $b$**，使模型能够平移
3. **送入激活函数**，以上的 $\sum(wx+b)$ 线性结构送入激活函数，引入非线性，使得神经网络能够逼近任意复杂函数

---

## 二、激活函数

### 1. Sigmoid 函数

**公式和导数：**
$$y = \frac{1}{1 + e^{-z}} \Rightarrow y' = y(1 - y)$$

![Sigmoid Activation Function](https://kimi-web-img.moonshot.cn/img/media.geeksforgeeks.org/c9aa699b2c86228a3229c3d968f1224e785329b1.png)

**特性：**
- $y$ 的取值范围是 $(0,1)$

**优点：**
- 简单、适用分类任务

**缺点：**
1. 反向传播训练时有**梯度消失**问题
2. 输出值区间为 $(0,1)$，关于 0 不对称
3. 梯度更新在不同方向走的太远，使得优化难度增大，训练耗时

---

### 2. Tanh 激活函数

**公式和导数：**
$$y = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} \Rightarrow y' = 1 - y^{2}$$

![Tanh Activation Function](https://kimi-web-img.moonshot.cn/img/media.geeksforgeeks.org/bf0728ee76da701e40baac9e9d7cb05564b2c535.png)

**优点：**
1. 解决了 Sigmoid 函数输出值非 0 对称的问题
2. 训练比 Sigmoid 函数快，更容易收敛

**缺点：**
1. 反向传播训练时有梯度消失问题
2. Tanh 函数和 Sigmoid 函数非常相似

---

### 3. ReLU 函数

**公式和导数：**
$$y = \begin{cases} 
z & \text{if } z > 0 \\ 
0 & \text{if } z \leq 0 
\end{cases} \Rightarrow y' = \begin{cases} 
1 & \text{if } z > 0 \\ 
0 & \text{if } z \leq 0 
\end{cases}$$

![ReLU Activation Function](https://kimi-web-img.moonshot.cn/img/media.geeksforgeeks.org/3a53d21d1983ef755ac1beae27bd41ccbb9e158b.png)

**优点：**
1. 解决了梯度消失的问题
2. 计算更简单，没有 Sigmoid 和 Tanh 函数的指数运算

**缺点：**
1. 训练时可能出现**神经元死亡**

---

### 4. Leaky ReLU 函数

**公式和导数：**
$$y = \begin{cases} 
z & \text{if } z > 0 \\ 
az & \text{if } z \leq 0 
\end{cases} \Rightarrow y' = \begin{cases} 
1 & \text{if } z > 0 \\ 
a & \text{if } z \leq 0 
\end{cases} \quad (a \neq 0 \text{或} 1)$$

![Leaky ReLU Activation Function](https://kimi-web-img.moonshot.cn/img/media.geeksforgeeks.org/22b52471c5a749f44b0ba226ff87b4757b119430.jpg)

**优点：**
1. 解决了 ReLU 神经元死亡问题

**缺点：**
1. 无法为正负输入值提供一致的关系预测（不同区间函数不同）

---

## 三、前向传播与训练

### 前向传播流程
输入 $x$ → 线性组合（权重·向量 + 偏置）→ 激活函数 → 输出预测值  

### 损失函数

**作用：** 利用损失函数，反向传播，更新 $w, b$

1. **均方误差（MSE）**：
   $$J(x) = \frac{1}{2m}\sum_{i=1}^{m}\left( f(x_i) - y_i \right)^{2}$$
   > （有个 $1/2$ 是为了让导数好算）

2. **交叉熵损失函数**（分类任务常用）

### 训练过程
前向传播 → 计算误差 → 反向传播（利用梯度下降，更新 w 和 b）



### 梯度下降

**参数更新公式：**

$$w = w - \alpha \frac{\partial J(w)}{\partial w}$$

$$b = b - \alpha \frac{\partial J(b)}{\partial b}$$

> 下降从负号体现，$\alpha$ 是超参数**学习率**

---

## 四、卷积神经网络（CNN）

### 图像在计算机中的本质

- CNN 的输入是图像，本质是**矩阵**
- 灰度由数值表示，红绿蓝也由数值表示
- **灰度图**：单通道
- **彩色图**：多通道（RGB）

### 全连接神经网络存在的问题

做分类/回归问题的时候，会破坏输入图像的**空间信息**：

> 它把输入展平成一维向量，完全忽略了数据的局部结构（如图像中相邻像素的关联、时序数据的顺序）。这既是"破坏输入结构"，但本质是**放弃了结构信息**，而不是物理上"破坏"了输入值。

### 卷积运算过程

**输入矩阵（特征图）× 卷积核（权重矩阵）= 输出特征图**

![Convolution Operation](https://kimi-web-img.moonshot.cn/img/maucher.home.hdm-stuttgart.de/ff6bb37a5f37ab2c29243b3cbbfc283cac6a8a4e.gif)

- **核（即 $w$）**：用梯度下降法更新，偏置 $b$ 也是（公式和之前一样）

#### 关键参数

| 参数 | 说明 | 影响 |
|------|------|------|
| **步幅（Stride）** | 滑动的步幅 | 步幅越大，输出矩阵越小 |
| **填充（Padding）** | 填充原本输出的矩阵 | 不填充的话，矩阵越变越小 |

**经过卷积运算后的特征图大小计算公式：**

$$\text{输出大小} = \left\lfloor \frac{\text{输入大小} - \text{核大小} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$$

> 填充操作为 $P$，步幅操作为 $S$。若结果为小数，一般向下取整。

#### 多通道数据卷积运算

- $F_N$ 个通道
- 用**宽度**来理解多通道
- 单通道没有宽度

### 池化操作

#### 最大池化（Max Pooling）
找某个区域的最大值

![Max Pooling](https://kimi-web-img.moonshot.cn/img/miro.medium.com/7a6416f41a5778a48fbb9793974a38a8bfa0decf.png)

#### 平均池化（Average Pooling）
对一个区域求平均值

> **注意：** 池化核不带未知参数（只是一个操作，不含网络参数）

**经过池化层后的特征图大小**，计算公式和卷积层一样。

---

## 五、卷积神经网络整体结构
输入层 → 卷积层 → 池化层 → [卷积层 → 池化层] × N → 全连接层 → 输出层

| 层级 | 英文 | 功能 |
|------|------|------|
| **输入层** | Input Layer | 接受原始图像数据/其他类型的网格结构数据 |
| **卷积层** | Convolution Layer | 通过卷积操作提取输入数据的局部特征。每个卷积核可以提取一种特定的特征，多个卷积核并行工作以提取不同类型的特征 |
| **池化层** | Pooling Layer | 对卷积层的输出进行下采样（降维），减少参数数量和提高计算效率。常见操作包括最大池化和平均池化 |
| **全连接层** | Fully Connected Layer | 将前面层提取的特征综合起来，用于分类/回归等任务。每个神经元都与前一层的所有神经元相连 |

---

## 六、总结对比表

| 激活函数 | 公式 | 输出范围 | 主要优点 | 主要缺点 |
|----------|------|----------|----------|----------|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $(0,1)$ | 简单，适合分类 | 梯度消失，非0对称 |
| **Tanh** | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $(-1,1)$ | 0对称，收敛快 | 仍有梯度消失 |
| **ReLU** | $\max(0,z)$ | $[0,+\infty)$ | 解决梯度消失，计算快 | 神经元死亡 |
| **Leaky ReLU** | $\max(az,z)$ | $(-\infty,+\infty)$ | 解决死亡问题 | 一致性差 |
