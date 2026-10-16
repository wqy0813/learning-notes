# KV Cache：原理、矩阵推导与推理流程

> Transformer 复习笔记 · Decoder-only · 从 Causal Self-Attention 推导 KV Cache

论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
官方文档：[Hugging Face：How caching works](https://huggingface.co/docs/transformers/cache_explanation) · [KV cache strategies](https://huggingface.co/docs/transformers/en/kv_cache)

> [!abstract] 核心思想
> KV Cache 是一种用于自回归推理的缓存机制。它利用 Causal Attention 中历史 Token 的表示不会被未来 Token 改变的性质，保存各层的历史 K、V，使后续生成不再重复计算旧位置的 Attention。

## 1. 主线概览

自回归生成每一步都依赖此前的上下文。如果每次重新计算整个前缀，就会重复处理历史位置。

**Causal Attention 保证旧位置无法读取未来信息**，所以新增 Token 不会改变旧位置的隐藏表示。由此可跳过旧位置的 Attention 计算：**只计算当前新位置，并复用各层历史 K/V**。

下面先用生成过程说明重复计算，再用 Score、Softmax 和 $A\times V$ 的矩阵推导解释为什么这项优化成立。

## 2. 为什么需要 KV Cache

### 2.1 自回归生成会重复处理前缀

Decoder-only 模型每次预测一个 Token，再把预测结果加入上下文。

交互：自回归生成逐步扩展上下文
```datacorejsx
const C = {
  border: '#d9d9d9',
  bg: '#ffffff',
  secondary: '#f5f5f5',
  text: '#333333',
  muted: '#666666',
  oldBg: '#f4f4f4', oldText: '#222222', oldLine: '#d8d8d8',
  newBg: '#bfeecb', newText: '#245a36', newLine: '#8bd6a1'
};
const STEPS = [
  { context: ['我', '喜欢'], next: '猫' },
  { context: ['我', '喜欢', '猫'], next: '和' },
  { context: ['我', '喜欢', '猫', '和'], next: '狗' }
];
function Chip({ value, predicted }) {
  return <span style={{
    display: 'inline-grid', placeItems: 'center', minWidth: '39px', padding: '5px 10px',
    borderRadius: '6px', whiteSpace: 'nowrap', fontWeight: 600,
    background: predicted ? C.newBg : C.oldBg,
    color: predicted ? C.newText : C.oldText,
    border: '1px solid ' + (predicted ? C.newLine : C.oldLine)
  }}>{value}</span>;
}
return function View() {
  const [step, setStep] = dc.useState(2);
  return <div style={{
    padding: '14px', borderRadius: '10px', border: '1px solid ' + C.border,
    color: C.text, background: C.bg, display: 'grid', gap: '12px'
  }}>
    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
      {STEPS.map((_, i) => <button key={i} type="button" aria-pressed={step === i}
        onClick={() => setStep(i)} style={{
          border: '1px solid ' + C.border, borderRadius: '6px', padding: '5px 12px',
          background: step === i ? '#eeeeee' : C.secondary,
          color: C.text,
          cursor: 'pointer'
        }}>第 {i + 1} 步</button>)}
      <span style={{ marginLeft: 'auto', fontSize: '0.85em', color: C.muted }}>灰：已有上下文 · 绿：本步预测</span>
    </div>
    <div style={{ overflowX: 'auto' }}>
      <div style={{ minWidth: '430px', display: 'grid', gap: '6px' }}>
        {STEPS.map((item, i) => <div key={i} style={{
          display: 'flex', gap: '7px', alignItems: 'center', borderRadius: '6px',
          padding: '8px', background: step === i ? '#fafafa' : 'transparent',
          borderLeft: '3px solid transparent',
          opacity: 1
        }}>
          <span style={{ width: '62px', flexShrink: 0, color: C.muted, fontSize: '0.9em', whiteSpace: 'nowrap' }}>第 {i + 1} 步</span>
          {item.context.map((token, j) => <Chip key={j} value={token} predicted={false} />)}
          <span style={{ color: C.muted }}>→</span>
          <Chip value={item.next} predicted={true} />
        </div>)}
      </div>
    </div>
    <div style={{ color: C.muted, fontSize: '0.9em' }}>当前选中第 {step + 1} 步：预测出“{STEPS[step].next}”。绿色 Token 将在下一轮才作为输入进入 Transformer。</div>
  </div>;
};
```

如果每一步都重新处理完整前缀，旧 Token 的 Q、K、V、Attention Score 和 Attention 输出都会被再次计算。

### 2.2 Causal Attention 使旧位置的结果可复用

位置 $i$ 只能关注位置 $1\sim i$。新 Token 加入后，旧位置能够读取的信息范围没有改变，所以旧位置的 Attention 输出可以复用。第 3 节将用矩阵说明这一点。

## 3. 从 Self-Attention 推导 KV Cache

为便于书写，先忽略 Batch 和 Head 维度，只讨论一个 Attention Head。设 Query 和 Key 的维度为 $d_k$，Value 的维度为 $d_v$。
### 3.1 从单个 Token 的 Q/K/V 出发

进入当前 Self-Attention 层后，第 $i$ 个 Token 根据该层的输入隐藏向量 $x_i$ 分别计算：
$$
q_i=x_iW_Q,\qquad k_i=x_iW_K,\qquad v_i=x_iW_V
$$
其中 $W_Q,W_K,W_V$ 是该层固定的投影矩阵，$q_i,k_i,v_i$ 分别是该 Token 的 Query、Key、Value。

### 3.2 新增 Token 后，Score 矩阵如何扩展？

假设原来有 $n$ 个 Token（位置 $0\sim n-1$），现在加入位置 $n$ 的新 Token。此时 Q、K 都新增一行，因此 Score 矩阵 $S=QK^T/\sqrt{d_k}$ 的形状自然从 $n\times n$ 扩展为 $(n+1)\times(n+1)$。**矩阵新增最后一行和最后一列。**

接着看**旧 Score 是否变化**：$Q$ 的第 $i$ 行与 $K^T$ 的第 $j$ 列相乘，等价于 $Q$ 的第 $i$ 行与 $K$ 的第 $j$ 行相乘。如果当前层旧 Token 的输入隐藏向量不变，那么旧 Q、K、V 不变，原有的 Score 也不变。此时，新列表示**旧 Q 与新 K** 的点积，新行表示**新 Q 与所有 K** 的点积。

**Causal Mask 的作用是保证这种“不变”能够传递到后续层。** 它在 Softmax 前屏蔽旧 Query 对新 Key 的分数，使旧位置的 Attention 权重与输出保持不变；于是下一层的旧位置输入也不变，下一层的旧 Q/K/V 就可以复用。逐层推导后，Decode 才能跳过历史位置，**只计算新 Token 对应的 Attention 行**（见第 4 节）。

下面用 $3\times3\rightarrow4\times4$ 的矩阵对比观察这一过程。

交互：对比新增 Token 前后的 3×3 与 4×4 Score 矩阵

```datacorejsx
const C = {
  border: 'var(--background-modifier-border)',
  bg: 'var(--background-primary)',
  neutral: 'var(--background-secondary)',
  text: 'var(--text-normal)',
  muted: 'var(--text-muted)',
  blueBg: '#e4eefc', blueText: '#255785', blueLine: '#92b9e8',
  orangeBg: '#fff0dc', orangeText: '#94601e', orangeLine: '#e3b775',
  greenBg: '#def3e7', greenText: '#216242', greenLine: '#86c7a1'
};
const STAGES = [
  { id: 'raw', title: '原始 Score', desc: '旧 Q/K 未变：右侧虚线框的 3×3 区域与左侧相同；仅新增最后一列和最后一行。' },
  { id: 'mask', title: 'Causal Mask', desc: '未来列被屏蔽为 −∞：前三个 Query 无法关注新加入的第 4 个 Key。' },
  { id: 'softmax', title: 'Softmax', desc: '被屏蔽的位置权重变成 0：旧行有效权重不变，只有第 4 行需要新计算。' },
  { id: 'decode', title: '只计算新行', desc: '实际 Decode 不构建整张 4×4 矩阵：只计算第 4 个 Query 的 1×4 Score。灰色区域无需重算。' }
];
function ScoreCell({ r, c, stage, size }) {
  const isNewRow = size === 4 && r === 4;
  const isNewCol = size === 4 && c === 4 && r < 4;
  const masked = c > r && stage !== 'raw' && stage !== 'decode';
  const skipped = stage === 'decode' && (!isNewRow);
  let symbol = stage === 'softmax' ? 'a' : 's';
  let value = masked ? (stage === 'softmax' ? '0' : '−∞') : skipped ? '—' : null;
  const bg = (masked || skipped) ? C.neutral : isNewRow ? C.greenBg : isNewCol ? C.orangeBg : C.blueBg;
  const fg = (masked || skipped) ? C.muted : isNewRow ? C.greenText : isNewCol ? C.orangeText : C.blueText;
  const border = (masked || skipped) ? C.border : isNewRow ? C.greenLine : isNewCol ? C.orangeLine : C.blueLine;
  return <div style={{
    height: '43px', width: '54px', display: 'grid', placeItems: 'center',
    borderRadius: '5px', background: bg, color: fg, fontWeight: isNewRow ? 650 : 500,
    border: '1px solid ' + border
  }}>{value === null ? <span>{symbol}<sub>{r}{c}</sub></span> : value}</div>;
}
function Matrix({ size, stage }) {
  const ids = Array.from({ length: size }, (_, i) => i + 1);
  const isRight = size === 4;
  return <div style={{ display: 'grid', gap: '9px', justifyItems: 'center', flexShrink: 0 }}>
    <div style={{ color: C.text, fontWeight: 650 }}>{isRight ? '新增后：4 × 4' : '新增前：3 × 3'}</div>
    <div style={{ position: 'relative' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '34px repeat(' + size + ', 54px)', gap: '4px', alignItems: 'center' }}>
        <div style={{ textAlign: 'center', fontSize: '0.85em', color: C.muted }}>Q / K</div>
        {ids.map(c => <div key={'h'+c} style={{ textAlign: 'center', color: C.muted, fontSize: '0.85em' }}>K<sub>{c}</sub></div>)}
        {ids.map(r => <Fragment key={'r'+r}>
          <div style={{ textAlign: 'center', color: C.muted, fontSize: '0.85em' }}>Q<sub>{r}</sub></div>
          {ids.map(c => <ScoreCell key={'cell'+r+'-'+c} r={r} c={c} stage={stage} size={size}/>)}
        </Fragment>)}
      </div>
      {isRight && <div aria-hidden="true" style={{
        position: 'absolute', left: '35px', top: '20px', width: '172px', height: '145px',
        border: '2px dashed ' + C.blueLine, borderRadius: '8px', pointerEvents: 'none'
      }} />}
    </div>
  </div>;
}
return function View() {
  const [stage, setStage] = dc.useState('raw');
  const info = STAGES.find(item => item.id === stage);
  return <div style={{
    border: '1px solid ' + C.border, borderRadius: '10px', background: C.bg,
    color: C.text, padding: '14px', display: 'grid', gap: '12px'
  }}>
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }} role="group" aria-label="矩阵计算阶段">
      {STAGES.map(item => <button type="button" key={item.id} aria-pressed={stage === item.id}
        onClick={() => setStage(item.id)} style={{
          borderRadius: '6px', padding: '6px 10px', cursor: 'pointer',
          border: '1px solid ' + C.border,
          background: stage === item.id ? 'var(--interactive-accent)' : C.neutral,
          color: stage === item.id ? 'var(--text-on-accent)' : C.text
        }}>{item.title}</button>)}
    </div>
    <div style={{ overflowX: 'auto', padding: '4px 0 8px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '26px', minWidth: '645px' }}>
        <Matrix size={3} stage={stage}/>
        <div style={{ color: C.muted, fontSize: '0.9em', textAlign: 'center', flexShrink: 0 }}>
          <div>新增</div><div>第 4 个 Token</div><div style={{ fontSize: '1.6em', lineHeight: '1.5' }}>→</div>
        </div>
        <Matrix size={4} stage={stage}/>
      </div>
    </div>
    <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', fontSize: '0.88em', color: C.muted }}>
      {[
        {bg:C.blueBg,line:C.blueLine,name:'历史 Score，不变'},
        {bg:C.orangeBg,line:C.orangeLine,name:'新增列'},
        {bg:C.greenBg,line:C.greenLine,name:'新增行'}
      ].map(item => <span key={item.name} style={{ display:'inline-flex',alignItems:'center',gap:'6px' }}>
        <span style={{ width:'12px',height:'12px',borderRadius:'3px',background:item.bg,border:'1px solid '+item.line }}/>{item.name}
      </span>)}
    </div>
    <div style={{ fontSize: '0.92em', color: C.text, paddingTop: '8px', borderTop: '1px solid ' + C.border }}>{info.desc}</div>
    <div style={{ color: C.muted, fontSize: '0.82em' }}>符号是示意，不代表具体数值；蓝色虚线框标识扩展后保留的旧 3×3 区域。</div>
  </div>;
};
```

### 3.3 Attention 输出只需要最后一行

Attention 输出为：
$$
O=AV
$$

假设有 4 个 Token，Value 向量维度 $d_v=3$，则矩阵形状为 $A\ (4\times4)$、$V\ (4\times3)$、$O\ (4\times3)$。

矩阵乘法中，**A 的第 $i$ 行 × V 的第 $j$ 列**，得到输出矩阵的元素 $o_{ij}$：
$$
o_{ij}=\sum_{k=1}^{4}a_{ik}v_{kj}
$$

由于旧行的 Attention 权重不变、旧 Value 不变，而新 Value 对旧行的权重为 $0$，因此旧位置的 Attention 输出也不变。只需计算新位置的输出：
$$
\boxed{o_4=A_4V}
$$

这里 $A_4$ 是 $1\times4$ 的最后一行权重，乘整个 $4\times d_v$ 的 V，得到 $1\times d_v$ 的新输出。下方交互图可以逐个查看 **A 的行 × V 的列 → O 的元素**。

**实际 Decode 不会先计算完整的 $O$ 再删掉旧行，而是直接跳过旧行。**

交互：选择 A 的行和 V 的列，查看对应的 O 元素

```datacorejsx
const BORDER = '1px solid #cfd9e6';
const CELL_BG = '#f3f6fa';
const ROW_BG = '#dce9ff';
const COL_BG = '#ffedd5';
const OUTPUT_BG = '#dff3e7';
const OUTPUT_ACTIVE_BG = '#8ed9a3';
const TEXT = '#34495e';
const A = [['a11', '0', '0', '0'], ['a21', 'a22', '0', '0'], ['a31', 'a32', 'a33', '0'], ['a41', 'a42', 'a43', 'a44']];
const V = [['v11', 'v12', 'v13'], ['v21', 'v22', 'v23'], ['v31', 'v32', 'v33'], ['v41', 'v42', 'v43']];

function SymbolCell({ value }) {
  return /^[av][1-4][1-4]$/.test(value) ? <span>{value[0]}<sub>{value.slice(1)}</sub></span> : value;
}

function Matrix({ data, kind, row, col }) {
  return <div style={{ display: 'grid', gridTemplateColumns: 'repeat(' + data[0].length + ', minmax(46px, 1fr))', gap: '4px' }}>
    {data.flatMap((line, r) => line.map((value, c) => {
      const active = kind === 'a' && r === row;
      const column = kind === 'v' && c === col;
      const output = kind === 'o' && r === row && c === col;
      const outputRow = kind === 'o' && r === 3;
      const background = output ? OUTPUT_ACTIVE_BG : outputRow ? OUTPUT_BG : active ? ROW_BG : column ? COL_BG : CELL_BG;
      return <div style={{ minHeight: '38px', display: 'grid', placeItems: 'center', border: BORDER, borderRadius: '5px', background }}>{kind === 'o' ? <span>o<sub>{r + 1}{c + 1}</sub></span> : <SymbolCell value={value} />}</div>;
    }))}
  </div>;
}

return function View() {
  const [row, setRow] = dc.useState(3);
  const [col, setCol] = dc.useState(0);
  const output = Array.from({ length: 4 }, (_, r) => Array.from({ length: 3 }, (_, c) => 'o' + (r + 1) + (c + 1)));
  const terms = A[row].map((a, i) => <>{i > 0 && ' + '}<SymbolCell value={a} />·<SymbolCell value={V[i][col]} /></>);
  return (
    <div style={{ border: BORDER, borderRadius: '8px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px', color: TEXT, background: '#ffffff' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
        <span>选择 A 的行：</span>{[0, 1, 2, 3].map(r => <button type="button" aria-pressed={row === r} onClick={() => setRow(r)}>{r + 1}</button>)}
        <span style={{ marginLeft: '8px' }}>选择 V 的列：</span>{[0, 1, 2].map(c => <button type="button" aria-pressed={col === c} onClick={() => setCol(c)}>{c + 1}</button>)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 1fr) 24px minmax(130px, 1fr) 24px minmax(130px, 1fr)', gap: '8px', alignItems: 'center' }}>
        <div><div style={{ textAlign: 'center', marginBottom: '5px' }}>A（4 × 4）</div><Matrix data={A} kind="a" row={row} col={col} /></div>
        <div style={{ textAlign: 'center', color: TEXT, fontSize: '1.4em' }}>×</div>
        <div><div style={{ textAlign: 'center', marginBottom: '5px' }}>V（4 × 3）</div><Matrix data={V} kind="v" row={row} col={col} /></div>
        <div style={{ textAlign: 'center', color: TEXT, fontSize: '1.4em' }}>=</div>
        <div><div style={{ textAlign: 'center', marginBottom: '5px' }}>O（4 × 3）</div><Matrix data={output} kind="o" row={row} col={col} /></div>
      </div>
      <div style={{ padding: '8px 10px', borderLeft: '3px solid #2563eb', background: '#f3f6fa' }}>
        <span>o<sub>{row + 1}{col + 1}</sub> = </span>{terms}
      </div>
    </div>
  );
};
```

### 3.4 新行需要哪些 Q、K、V？

只计算第 4 个 Token 的 Attention 时：
$$
\boxed{
o_4=\operatorname{softmax}
\left(\frac{q_4K^T}{\sqrt{d_k}}\right)V
}
$$
计算顺序是：**新 $q_4$ 与全部 K 点积 → 缩放、Softmax 得到一行权重 → 与全部 V 加权求和，得到 $o_4$**。
此时 $q_4K^T$ 的 Score 形状是 $1\times4$，而不再需要计算完整的 $4\times4$ Score。

| 张量 | 是否需要历史部分 | 原因 |
| --- | --- | --- |
| $Q$ | 不需要 | 当前步骤只使用新 Query |
| $K$ | 需要 | 新 Query 要与所有历史 Key 计算匹配分数 |
| $V$ | 需要 | 输出要聚合所有历史 Value |

**所以历史 Q 不需要缓存；历史 K/V 会被后续新 Token 反复读取，应该缓存。**

## 4. 为什么每个 Transformer Block 都能复用历史 K/V

第 3 节已经证明：**新增 Token 后，由于 Causal Mask 的存在，旧 Token 的 Attention 输出不变。**

但 Transformer 通常由多个 Block 堆叠而成，**上一层 Block 的输出，就是下一层 Block 的输入**。因此，还需要证明旧 Token 的表示在每一层都不变。

假设原序列为 [A, B]，现在新增 Token C，以 Pre-LN Transformer 为例。
![[Pasted image 20261016162232.png]]

### ① 第一层：旧 Token 的 Block 输出不变

新增 C 后，A、B 的输入表示不变。由于 Causal Mask，A、B 无法关注 C，因此它们的 Attention 输出仍然不变。记第 $i$ 个位置的 Attention 输出为 $o_i$，第一次残差连接为：
$$
u_i=x_i+o_i
$$

对于旧 Token，新增 C 前后 $x_i$、$o_i$ 都不变，因此：
$$
x_i\text{ 不变},\qquad o_i\text{ 不变}
$$
$$
\boxed{u_i\text{ 不变}}
$$

接下来进入第二个 Pre-LN 子层。LayerNorm 只在当前 Token 自己的 $D$ 个特征上计算均值和方差，不跨 Token 读取其他位置。其逐 Token 计算链为：
$$
\begin{aligned}
z_i&=\operatorname{Norm}(u_i),\\
\operatorname{FFN}(z_i)&=W_2\,\sigma(W_1z_i+b_1)+b_2,\\
h_i&=u_i+\operatorname{FFN}(z_i).
\end{aligned}
$$

由于 $u_i$ 不变、模型参数固定，$z_i$ 和 $\operatorname{FFN}(z_i)$ 也不变，因此：
$$
\boxed{h_i\text{ 不变}}
$$

也就是说，**旧 Token 经过第一层完整 Block 后的最终输出仍然不变。**
### ② 第二层：旧 Token 的 K/V 也不变

关键在于：
**第一层的输出 $H_1$，就是第二层的输入。**
因此第一层中旧 Token 的输出不变，就意味着第二层中旧 Token 的输入也不变。

第二层 Pre-LN 先归一化，再使用第二层自己的投影矩阵计算 K/V：
$$
\widetilde{H}_1=\operatorname{Norm}(H_1),\qquad
\boxed{K^{(2)}=\widetilde{H}_1W_K^{(2)}},\qquad
\boxed{V^{(2)}=\widetilde{H}_1W_V^{(2)}}
$$
其中 $W_K^{(2)}$、$W_V^{(2)}$ 是**第二层自己的投影矩阵**。

由于旧 Token 的 $H_1$ 不变，因此：
$$
\operatorname{Norm}(H_1)\text{ 不变}
$$
进而：
$$
\boxed{
K^{(2)}_{\text{旧}}\text{ 不变}
}
$$
$$
\boxed{
V^{(2)}_{\text{旧}}\text{ 不变}
}
$$
所以第二层旧 Token 的 K/V 可以直接从第二层自己的 KV Cache 中复用。

同时，第二层同样满足 Causal Attention、逐位置 LayerNorm、FFN 和残差连接，因此第二层旧 Token 的最终输出也不变。
### ③ 逐层推广
于是形成一条完整的传递链：
$$
\boxed{\text{旧输入不变}}
\;\downarrow\;
\boxed{\text{Causal Attention：旧位置输出不变}}
\;\downarrow\;
\boxed{\text{LayerNorm / FFN / Residual：旧位置继续不变}}
\;\downarrow\;
\boxed{\text{第 1 层旧输出不变}}
\;\downarrow\;
\boxed{\text{第 2 层旧输入不变}}
\;\downarrow\;
\boxed{\text{第 2 层旧 K/V 不变}}
$$
继续向后传递……

因此，**所有 Transformer Block 都可以复用历史 Token 的 K/V，只需要计算新 Token 对应的 K/V。**

不过，每个 Block 的输入和投影矩阵不同，因此每层计算出的 K/V 也不同：
$$
\left(K^{(1)},V^{(1)}\right),\quad
\left(K^{(2)},V^{(2)}\right),\quad
\left(K^{(3)},V^{(3)}\right),\quad\ldots
$$
所以，**每个 Block 都必须维护自己独立的 KV Cache，不能直接共用其他层的缓存。**

> **适用前提：** 模型参数和位置索引保持一致，使用因果注意力，LayerNorm、FFN、残差等操作逐 Token 位置计算。对于 RoPE，复用缓存时还必须保持历史 Token 原来的位置索引。上述结论适用于标准因果 Decoder；如果结构包含非因果的跨位置操作，则不能直接套用。

## 5. GPT 推理中的两个阶段

### 5.1 Prefill：建立初始 KV Cache

假设 Prompt 是：

```text
我 喜欢 猫
```

Prefill 阶段一次处理整个 Prompt：

1. 每个 Block 为所有 Prompt 位置计算 K 和 V。
2. 将各 Block 的 K/V 写入对应缓存。
3. 使用最后一个位置的隐藏状态，经 LM Head 预测下一个 Token，例如“和”。

刚预测出的“和”还没有经过下一次 Transformer 前向计算，因此它的 K/V 此时还没有写入缓存。

### 5.2 Decode：逐 Token 读取并追加缓存

下一步把“和”作为新输入：

1. 当前输入只包含这个新 Token。
2. 每个 Block 只计算当前 Token 的 $q_t,k_t,v_t$。
3. 每层将当前 $k_t,v_t$ 与该层历史 K/V 合并，并更新本层缓存。
4. 当前 $q_t$ 与合并后的全部 K/V 计算 Attention。
5. 经过后续 Block、最终 Norm 和 LM Head，预测下一个 Token。

Hugging Face 的缓存接口也采用“用当前 K/V 更新缓存，再用更新后的 K/V 计算 Attention”的流程。[来源](https://huggingface.co/docs/transformers/cache_explanation)

交互：观察 Prefill 与 Decode 时 KV Cache 如何增长

```datacorejsx
const C = {
  border: 'var(--background-modifier-border)',
  bg: 'var(--background-primary)',
  panel: 'var(--background-secondary)',
  text: 'var(--text-normal)', muted: 'var(--text-muted)',
  oldBg: '#e8f1fc', oldText: '#24547d', oldLine: '#a5c6e6',
  newBg: '#e1f3e8', newText: '#216044', newLine: '#94d0ad'
};
const STEPS = [
  { label: '① Prefill', phase: 'Prefill', input: ['我','喜欢','猫'], before: [],
    predicted: '和', shape: '3 × 3', q: 3, tip: '一次处理整个 Prompt；本次产生三个位置的 K/V，并存入缓存。预测出的“和”还未计算 K/V。' },
  { label: '② Decode 1', phase: 'Decode', input: ['和'], before: ['我','喜欢','猫'],
    predicted: '狗', shape: '1 × 4', q: 1, tip: '仅输入“和”，计算它的 Q/K/V；使用前三个位置的历史缓存，再把“和”的 K/V 加入缓存。' },
  { label: '③ Decode 2', phase: 'Decode', input: ['狗'], before: ['我','喜欢','猫','和'],
    predicted: '。', shape: '1 × 5', q: 1, tip: '仅输入“狗”，复用前四个位置的 K/V。刚预测出的“。”要到下一轮才计算对应 K/V。' }
];
function Chip({ token, fresh, faded }) {
  return <span style={{
    display: 'inline-grid', placeItems: 'center', minWidth: '49px',
    padding: '7px 9px', borderRadius: '6px', fontWeight: 600,
    background: faded ? C.panel : fresh ? C.newBg : C.oldBg,
    color: faded ? C.muted : fresh ? C.newText : C.oldText,
    border: '1px ' + (faded ? 'dashed' : 'solid') + ' ' + (faded ? C.border : fresh ? C.newLine : C.oldLine)
  }}>{token}</span>;
}
function Tokens({ values, fresh, empty }) {
  return <div style={{ display:'flex',flexWrap:'wrap',gap:'5px',alignItems:'center',minHeight:'39px' }}>
    {values.length ? values.map((t,i) => <Chip key={i} token={t} fresh={fresh} />) :
     <span style={{ color: C.muted, fontSize:'0.9em' }}>{empty || '无'}</span>}
  </div>;
}
return function View() {
  const [index, setIndex] = dc.useState(0);
  const item = STEPS[index];
  const after = item.before.concat(item.input);
  return <div style={{
    display:'grid',gap:'12px',padding:'14px',border:'1px solid '+C.border,
    borderRadius:'10px',color:C.text,background:C.bg
  }}>
    <div style={{ display:'flex',gap:'6px',flexWrap:'wrap' }} role="group" aria-label="推理阶段">
      {STEPS.map((s,i) => <button key={i} type="button" aria-pressed={index===i}
        onClick={() => setIndex(i)} style={{
          cursor:'pointer',padding:'6px 12px',borderRadius:'6px',
          border:'1px solid '+C.border,
          background:index===i?'var(--interactive-accent)':C.panel,
          color:index===i?'var(--text-on-accent)':C.text
        }}>{s.label}</button>)}
    </div>
    <div style={{ display:'grid',gap:'10px',gridTemplateColumns:'minmax(0,1fr) minmax(0,1fr)' }}>
      <div style={{ minWidth:0,border:'1px solid '+C.border,borderRadius:'8px',padding:'10px',background:C.panel }}>
        <div style={{ color:C.muted,fontSize:'0.88em',marginBottom:'6px' }}>本次送入 Transformer</div>
        <Tokens values={item.input} fresh={true}/>
      </div>
      <div style={{ minWidth:0,border:'1px solid '+C.border,borderRadius:'8px',padding:'10px',background:C.panel }}>
        <div style={{ color:C.muted,fontSize:'0.88em',marginBottom:'6px' }}>Attention Score（单 Head）</div>
        <strong>{item.shape}</strong><span style={{color:C.muted,fontSize:'0.85em'}}> · Query {item.q} 行</span>
      </div>
    </div>
    <div style={{ borderTop:'1px solid '+C.border,paddingTop:'10px',display:'grid',gap:'10px' }}>
      <div style={{ display:'grid',gap:'5px' }}>
        <div style={{ color:C.muted,fontSize:'0.9em' }}>计算前已有缓存的 K/V（蓝）</div>
        <Tokens values={item.before} fresh={false} empty="空：Prefill 是首次处理 Prompt"/>
      </div>
      <div style={{ display:'grid',gap:'5px' }}>
        <div style={{ color:C.muted,fontSize:'0.9em' }}>本次新计算的 K/V（绿）</div>
        <Tokens values={item.input} fresh={true}/>
      </div>
      <div style={{ display:'flex',gap:'8px',alignItems:'center',color:C.muted,fontSize:'0.9em' }}>
        <span>历史 K/V + 本次 K/V</span><span>↓</span><span>供当前 Attention 使用，并写入缓存</span>
      </div>
      <div style={{ display:'grid',gap:'5px',padding:'10px',border:'1px solid '+C.border,borderRadius:'8px' }}>
        <strong>本轮结束后的 KV Cache（{after.length} 个 Token）</strong>
        <div style={{ display:'flex',gap:'5px',flexWrap:'wrap' }}>
          {after.map((t,i) => <Chip key={i} token={t} fresh={i>=item.before.length}/>)}
        </div>
      </div>
    </div>
    <div style={{display:'flex',gap:'8px',alignItems:'center',flexWrap:'wrap',borderTop:'1px solid '+C.border,paddingTop:'10px'}}>
      <span style={{fontSize:'0.9em',color:C.muted}}>LM Head 本步预测：</span>
      <Chip token={item.predicted} faded={true}/>
      <span style={{fontSize:'0.86em',color:C.muted}}>尚未加入 KV Cache；下一轮才作为输入</span>
    </div>
    <div style={{fontSize:'0.9em',color:C.text}}>{item.tip}</div>
    <div style={{fontSize:'0.83em',color:C.muted}}>示意：只画一个 Transformer Block 的 KV；真实模型的每一层分别维护缓存。</div>
  </div>;
};
```

### 5.3 Prefill 与 Decode 对比

| 项目 | Prefill | Decode |
| --- | --- | --- |
| 输入 | 整个 Prompt | 新生成的 Token |
| Q/K/V 计算 | 所有输入位置 | 当前新位置 |
| Attention Score | 通常为 $N\times N$ | 通常为 $1\times T$ |
| KV Cache | 建立初始缓存 | 读取并追加 |
| 主要任务 | 编码已有上下文 | 逐 Token 生成 |

这里的 $N$ 是 Prompt 长度，$T$ 是加入当前 Token 后的缓存总长度；表格忽略 Batch 和 Head 维度。

## 6. KV Cache 的形状与存储

### 6.1 每个 Block 都有独立缓存

不同 Block 的输入隐藏状态和投影矩阵不同，因此它们产生的 K/V 不同：

```text
Transformer
├── Block 1 → K_cache_1, V_cache_1
├── Block 2 → K_cache_2, V_cache_2
├── Block 3 → K_cache_3, V_cache_3
│     ⋮
└── Block L → K_cache_L, V_cache_L
```

某一层的常见逻辑形状为：

$$
K_{\text{cache}},V_{\text{cache}}
\in\mathbb{R}^{B\times H_{kv}\times T\times d}
$$

| 符号 | 含义 |
| --- | --- |
| $B$ | Batch Size |
| $H_{kv}$ | KV Head 数量 |
| $T$ | 已缓存的 Token 数量 |
| $d$ | 每个 KV Head 的维度 |

在 K 和 V 的 Head 维度相同的简化条件下，每加入一个 Token，逻辑形状从：

$$
[B,H_{kv},T,d]\longrightarrow[B,H_{kv},T+1,d]
$$

实际实现可以预分配固定容量，或按块管理缓存，不一定每一步都复制并拼接整个张量。

### 6.2 显存开销

每层都保存历史 K 和 V，所以缓存量随 $T$ 线性增长。忽略元数据和对齐开销时：

$$
\boxed{M_{\text{KV}}=2LBH_{kv}Td\,b}
$$

其中：

- $2$：同时保存 K 和 V；
- $L$：Transformer Block 层数；
- $B$：Batch Size；
- $H_{kv}$：KV Head 数量；
- $T$：缓存长度；
- $d$：每个 Head 的维度；
- $b$：每个元素占用的字节数，例如 FP16 为 2 字节。

因此，长上下文、大 Batch 和高并发会增加 KV Cache 的显存需求。当前 Transformers 文档也将 KV Cache 占用大量内存列为长上下文生成中的主要限制之一。[来源](https://huggingface.co/docs/transformers/en/kv_cache)

## 7. MHA、GQA 与 MQA

三种结构的核心差异是 $H_{kv}$ 的数量：

| 结构 | K/V 组织方式 | 对 KV Cache 的影响 |
| --- | --- | --- |
| MHA（Multi-Head Attention） | 每个 Query Head 都有对应的 K/V Head | $H_{kv}$ 较大，缓存较大 |
| GQA（Grouped-Query Attention） | 多个 Query Head 共享一组 K/V Head | 减少 $H_{kv}$，缓存小于 MHA |
| MQA（Multi-Query Attention） | 所有 Query Head 共享一组 K/V | $H_{kv}=1$，缓存最小 |

三者都复用历史 K/V；变化的是缓存中需要保存的 KV Head 数量。

## 8. KV Cache 节省了什么

KV Cache 同时减少两类重复工作：

1. 历史位置的 K/V 投影不再重复计算。
2. 旧位置的 Attention 输出及其后续 Block 计算不再重复计算。

| 项目 | 不使用 KV Cache | 使用 KV Cache |
| --- | --- | --- |
| 每步输入 | 完整前缀 | 新 Token |
| Q/K/V 投影 | 重新计算所有位置 | 只计算新位置 |
| 当前 Attention Score | 约为 $T\times T$ | 约为 $1\times T$ |
| 旧位置输出 | 重复计算 | 跳过 |
| 持久 KV 显存 | 不需要 | 随上下文长度增长 |

单步 Attention 仍然需要新 Query 与所有历史 Key 匹配：

$$
q_{\text{new}}K_{\text{all}}^T
$$

因此，对于一个 Head，忽略常数和其他算子：

- 重新计算完整前缀：$O(T^2d)$；
- 使用 KV Cache 计算新位置：$O(Td)$。

这是 Attention 主体的算术量比较。实际吞吐还取决于 KV 读取带宽、内存访问、算子实现和硬件。

## 9. 三个常见问题

### 9.1 KV Cache 用于训练吗？

通常不以逐 Token 增量缓存的方式用于标准全序列训练。训练需要同时计算多个位置并反向传播，更适合并行处理；KV Cache 主要服务于自回归推理。[来源](https://huggingface.co/docs/transformers/cache_explanation)

### 9.2 KV Cache 保存完整历史隐藏向量吗？

不是。核心内容是每个 Attention 层的 K/V 张量，不是所有旧 Token 的完整隐藏向量。运行时可能存在其他缓冲区，但它们不属于 KV Cache 的核心定义。

### 9.3 KV Cache 可以无限增长吗？

不能。缓存受显存、最大上下文长度和模型结构限制。工程实现可以采用固定大小缓存、滑动窗口、量化、分页或其他缓存管理策略；具体能力取决于推理框架和模型。[来源](https://huggingface.co/docs/transformers/en/kv_cache)

## 来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017，原始 Transformer 论文，访问于 2026-10-15。
- [How caching works](https://huggingface.co/docs/transformers/cache_explanation) — Hugging Face Transformers 官方文档，访问于 2026-10-15。
- [KV cache strategies](https://huggingface.co/docs/transformers/en/kv_cache) — Hugging Face Transformers 官方文档，访问于 2026-10-15。

资料核对日期：2026-10-15

