# 1. Linear Algebra — the Language of Neural Networks

## 1.1 Vectors

This Qiita exercise page follows one rule:

```text
one vector concept → one deep learning / machine learning anchor → one small exercise → NumPy + PyTorch check
```

The goal is not to memorize vector definitions.

The goal is to make vector math usable for neural networks, embeddings, attention, retrieval, gradients, and representation learning.

---

## Concept map

| Vector concept | Deep learning / machine learning anchor |
|---|---|
| Vectors as ordered lists | Feature vectors, embeddings, hidden states |
| Row vectors vs column vectors | Shape correctness in dot products and linear layers |
| Vector addition | Residual connections, token + positional embeddings |
| Scalar multiplication | Learning-rate-scaled gradients, attention scaling |
| Linear combinations | Neurons compute weighted sums |
| Span | What representations a model can express |
| Basis | Coordinate systems, embedding spaces, PCA |
| Dimension | Feature size, latent size, hidden size, `d_model` |
| Norms | Weight decay, gradient clipping, embedding magnitude |
| Unit vectors | Direction-only comparison, normalized embeddings |
| Distance between vectors | Retrieval, nearest neighbors, clustering |
| Dot product | Linear layers, attention scores, similarity |
| Orthogonality | Decorrelation, PCA, independent representation directions |
| Projection | PCA, least squares, removing / isolating representation directions |
| Cosine similarity | Semantic search, contrastive learning, recommendation |

---

<details>
<summary><strong>Q1. Vectors as ordered lists</strong> — feature vectors, embeddings, hidden states</summary>

---

### 🧠 Concept

A vector is an ordered list of numbers.

In deep learning, one vector can represent one input example, one token embedding, one hidden state, one item embedding, or one gradient.

| Vector type | Deep learning meaning |
|---|---|
| Feature vector | One row of tabular input |
| Token embedding | Learned vector for one token |
| Hidden state | Internal representation inside a network |
| Gradient vector | Update direction for parameters |

The order matters because each position has a fixed meaning.

---

### 🧮 Formula

```math
x =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
```

Order-sensitive representation:

```math
\begin{bmatrix}
0.8 \\
0.1 \\
-0.4
\end{bmatrix}
\neq
\begin{bmatrix}
-0.4 \\
0.8 \\
0.1
\end{bmatrix}
```

---

### 🎯 Problem

A model receives this input feature vector:

```math
x =
\begin{bmatrix}
0.2 \\
0.5 \\
-1.0 \\
3.0
\end{bmatrix}
```

Answer:

1. How many features does this input have?
2. Why does the order matter for a neural network?

---

<details>
<summary><strong>✅ Show answer</strong></summary>

The vector has:

```math
4
```

features.

Each position has a fixed meaning.

For example:

| Position | Possible meaning |
|---:|---|
| `x_1` | normalized age |
| `x_2` | normalized income |
| `x_3` | risk score |
| `x_4` | event count |

If the order changes, the model multiplies the wrong feature by the wrong weight.

Deep learning interpretation:

A neural network does not see feature names.

It sees positions.

So the vector order is part of the model input contract.

</details>

---

### 💻 Code task

Create the vector and inspect its shape.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([0.2, 0.5, -1.0, 3.0])

print(x)
print("shape:", x.shape)
print("number of features:", x.shape[0])
```

Output:

```python
[ 0.2  0.5 -1.   3. ]
shape: (4,)
number of features: 4
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([0.2, 0.5, -1.0, 3.0])

print(x)
print("shape:", x.shape)
print("number of features:", x.shape[0])
```

Output:

```python
tensor([ 0.2000,  0.5000, -1.0000,  3.0000])
shape: torch.Size([4])
number of features: 4
```

</details>

---

### ⚠️ Common mistake

A vector is not an unordered bag of numbers.

Changing the order changes the meaning of the input.

</details>

---

<details>
<summary><strong>Q2. Row vectors vs column vectors</strong> — linear layer shape logic</summary>

---

### 🧠 Concept

A row vector is horizontal.

```math
x =
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
```

A column vector is vertical.

```math
w =
\begin{bmatrix}
0.1 \\
0.2 \\
0.3
\end{bmatrix}
```

In deep learning, this matters because matrix multiplication depends on shape.

A single input example can be treated as a row vector.

A single neuron weight can be treated as a column vector.

Then:

```math
z = xw + b
```

---

### 🧮 Formula

Shape logic:

```math
(1 \times d)(d \times 1) = (1 \times 1)
```

Concrete example:

```math
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
0.1 \\
0.2 \\
0.3
\end{bmatrix}
=
1(0.1) + 2(0.2) + 3(0.3)
```

---

### 🎯 Problem

Given:

```math
x =
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
```

```math
w =
\begin{bmatrix}
0.1 \\
0.2 \\
0.3
\end{bmatrix}
```

```math
b = 0.5
```

Compute:

```math
z = xw + b
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

First compute:

```math
xw = 1(0.1) + 2(0.2) + 3(0.3)
```

```math
xw = 0.1 + 0.4 + 0.9
```

```math
xw = 1.4
```

Then add bias:

```math
z = 1.4 + 0.5
```

```math
z = 1.9
```

Deep learning interpretation:

This is one neuron's pre-activation value.

For one example and one neuron:

```math
(1 \times d)(d \times 1) \rightarrow (1 \times 1)
```

For a batch and one neuron:

```math
(N \times d)(d \times 1) \rightarrow (N \times 1)
```

</details>

---

### 💻 Code task

Compute one-neuron output with explicit row and column shapes.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([[1.0, 2.0, 3.0]])      # shape: (1, 3)
w = np.array([[0.1], [0.2], [0.3]])  # shape: (3, 1)
b = 0.5

z = x @ w + b

print("z =", z)
print("z shape:", z.shape)
```

Output:

```python
z = [[1.9]]
z shape: (1, 1)
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0]])      # shape: (1, 3)
w = torch.tensor([[0.1], [0.2], [0.3]])  # shape: (3, 1)
b = torch.tensor(0.5)

z = x @ w + b

print("z =", z)
print("z shape:", z.shape)
```

Output:

```python
z = tensor([[1.9000]])
z shape: torch.Size([1, 1])
```

</details>

---

### ⚠️ Common mistake

Row/column vector shape logic is not mainly about element-wise multiplication.

Element-wise multiplication keeps component-wise products:

```math
x \odot w
```

A neuron uses multiplication and summation:

```math
w \cdot x + b
```

</details>

---

<details>
<summary><strong>Q3. Vector addition</strong> — residual connections and positional embeddings</summary>

---

### 🧠 Concept

Vector addition combines vectors component by component.

In deep learning, vector addition appears in:

| Use case | Meaning |
|---|---|
| token embedding + positional embedding | add word meaning and position information |
| residual connection | add original representation back to transformed representation |
| multimodal fusion | combine representations from different sources |

---

### 🧮 Formula

```math
a + b =
\begin{bmatrix}
a_1 + b_1 \\
a_2 + b_2 \\
a_3 + b_3
\end{bmatrix}
```

---

### 🎯 Problem

A token embedding is:

```math
e =
\begin{bmatrix}
0.20 \\
0.50 \\
-0.10
\end{bmatrix}
```

A positional embedding is:

```math
p =
\begin{bmatrix}
0.01 \\
-0.02 \\
0.03
\end{bmatrix}
```

Compute the combined representation:

```math
h = e + p
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
h =
\begin{bmatrix}
0.20 + 0.01 \\
0.50 + (-0.02) \\
-0.10 + 0.03
\end{bmatrix}
```

```math
h =
\begin{bmatrix}
0.21 \\
0.48 \\
-0.07
\end{bmatrix}
```

Deep learning interpretation:

Transformers often combine token meaning and token position by vector addition.

Both vectors must have the same dimension.

</details>

---

### 💻 Code task

Add token and positional embeddings.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

token_embedding = np.array([0.20, 0.50, -0.10])
position_embedding = np.array([0.01, -0.02, 0.03])

h = token_embedding + position_embedding

print(h)
```

Output:

```python
[ 0.21  0.48 -0.07]
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

token_embedding = torch.tensor([0.20, 0.50, -0.10])
position_embedding = torch.tensor([0.01, -0.02, 0.03])

h = token_embedding + position_embedding

print(h)
```

Output:

```python
tensor([ 0.2100,  0.4800, -0.0700])
```

</details>

---

### ⚠️ Common mistake

Vector addition requires matching shapes.

You cannot directly add a 768-dimensional embedding to a 512-dimensional embedding.

</details>

---

<details>
<summary><strong>Q4. Scalar multiplication</strong> — learning-rate-scaled gradients</summary>

---

### 🧠 Concept

Scalar multiplication multiplies every vector component by the same number.

In deep learning, this appears directly in gradient descent:

```math
\theta_{\text{new}} = \theta_{\text{old}} - \eta g
```

where:

| Symbol | Meaning |
|---|---|
| `\theta` | parameter vector |
| `g` | gradient vector |
| `\eta` | learning rate |

The learning rate is a scalar.

The gradient is a vector.

---

### 🧮 Formula

```math
cg =
\begin{bmatrix}
cg_1 \\
cg_2 \\
cg_3
\end{bmatrix}
```

---

### 🎯 Problem

A gradient vector is:

```math
g =
\begin{bmatrix}
4 \\
-2 \\
1
\end{bmatrix}
```

The learning rate is:

```math
\eta = 0.1
```

Compute:

```math
\eta g
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
\eta g =
0.1
\begin{bmatrix}
4 \\
-2 \\
1
\end{bmatrix}
```

```math
\eta g =
\begin{bmatrix}
0.4 \\
-0.2 \\
0.1
\end{bmatrix}
```

Deep learning interpretation:

Gradient descent does not subtract the full gradient.

It subtracts the learning-rate-scaled gradient.

</details>

---

### 💻 Code task

Scale a gradient vector.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

g = np.array([4.0, -2.0, 1.0])
lr = 0.1

step = lr * g

print(step)
```

Output:

```python
[ 0.4 -0.2  0.1]
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

g = torch.tensor([4.0, -2.0, 1.0])
lr = 0.1

step = lr * g

print(step)
```

Output:

```python
tensor([ 0.4000, -0.2000,  0.1000])
```

</details>

---

### ⚠️ Common mistake

A learning rate that is too large makes the update too aggressive.

That is not a small implementation detail.

It can break training.

</details>

---

<details>
<summary><strong>Q5. Linear combinations</strong> — neurons compute weighted sums</summary>

---

### 🧠 Concept

A linear combination multiplies values by weights and adds the results.

A neuron before activation is a linear combination of input features plus bias:

```math
z = w_1x_1 + w_2x_2 + \cdots + w_dx_d + b
```

This is the basic unit of a neural network.

---

### 🧮 Formula

```math
w \cdot x + b =
w_1x_1 + w_2x_2 + w_3x_3 + b
```

---

### 🎯 Problem

Given:

```math
x =
\begin{bmatrix}
2 \\
-1 \\
3
\end{bmatrix}
```

```math
w =
\begin{bmatrix}
0.5 \\
2 \\
-1
\end{bmatrix}
```

```math
b = 0.1
```

Compute:

```math
z = w \cdot x + b
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
z = (0.5)(2) + (2)(-1) + (-1)(3) + 0.1
```

```math
z = 1 - 2 - 3 + 0.1
```

```math
z = -3.9
```

Deep learning interpretation:

This is one neuron's pre-activation value.

If the activation is ReLU:

```math
\text{ReLU}(-3.9) = 0
```

</details>

---

### 💻 Code task

Compute one neuron weighted sum.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([2.0, -1.0, 3.0])
w = np.array([0.5, 2.0, -1.0])
b = 0.1

z = np.dot(w, x) + b

print(z)
```

Output:

```python
-3.9
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([2.0, -1.0, 3.0])
w = torch.tensor([0.5, 2.0, -1.0])
b = torch.tensor(0.1)

z = torch.dot(w, x) + b

print(z)
```

Output:

```python
tensor(-3.9000)
```

</details>

---

### ⚠️ Common mistake

A neuron is not mysterious.

Before the activation function, it is just a weighted sum.

</details>

---

<details>
<summary><strong>Q6. Span</strong> — what representations can be expressed</summary>

---

### 🧠 Concept

The span of vectors is the set of all vectors that can be created by linear combinations of them.

In machine learning, span answers:

```text
What outputs or representations can this set of features produce?
```

If a target vector is outside the span of available feature columns, a linear model cannot represent it exactly.

---

### 🧮 Formula

```math
\text{span}(v_1, v_2)
=
\{c_1v_1 + c_2v_2 \mid c_1,c_2 \in \mathbb{R}\}
```

---

### 🎯 Problem

Given representation directions:

```math
v_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

```math
v_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

Can they express this target representation?

```math
y =
\begin{bmatrix}
3 \\
-2
\end{bmatrix}
```

Find:

```math
c_1,\quad c_2
```

such that:

```math
c_1v_1 + c_2v_2 = y
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
c_1
\begin{bmatrix}
1 \\
0
\end{bmatrix}
+
c_2
\begin{bmatrix}
0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
c_1 \\
c_2
\end{bmatrix}
```

We need:

```math
\begin{bmatrix}
c_1 \\
c_2
\end{bmatrix}
=
\begin{bmatrix}
3 \\
-2
\end{bmatrix}
```

So:

```math
c_1 = 3,\quad c_2 = -2
```

Deep learning interpretation:

If learned representation directions span the needed space, the model can build the target representation.

If they do not, the model is expressively limited.

</details>

---

### 💻 Code task

Build the target vector from two representation directions.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 1.0])

c1 = 3.0
c2 = -2.0

y = c1 * v1 + c2 * v2

print(y)
```

Output:

```python
[ 3. -2.]
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

v1 = torch.tensor([1.0, 0.0])
v2 = torch.tensor([0.0, 1.0])

c1 = 3.0
c2 = -2.0

y = c1 * v1 + c2 * v2

print(y)
```

Output:

```python
tensor([ 3., -2.])
```

</details>

---

### ⚠️ Common mistake

Span is not a single vector.

Span is the whole set of vectors reachable by linear combinations.

</details>

---

<details>
<summary><strong>Q7. Basis</strong> — coordinate systems, embedding spaces, PCA</summary>

---

### 🧠 Concept

A basis is a set of vectors that can represent every vector in a space uniquely.

In machine learning, basis thinking appears in:

| Use case | Meaning |
|---|---|
| embedding dimensions | coordinates in a learned representation space |
| PCA | new orthogonal basis directions of maximum variance |
| latent spaces | coordinates for compressed representations |
| LoRA | low-rank update directions |

A vector's numbers are coordinates relative to a basis.

---

### 🧮 Formula

Standard basis in 2D:

```math
e_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
e_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

Any vector:

```math
x =
\begin{bmatrix}
a \\
b
\end{bmatrix}
```

can be written as:

```math
x = ae_1 + be_2
```

---

### 🎯 Problem

Write this vector using the standard basis:

```math
x =
\begin{bmatrix}
4 \\
-3
\end{bmatrix}
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
x =
4
\begin{bmatrix}
1 \\
0
\end{bmatrix}
+
(-3)
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

So:

```math
x = 4e_1 - 3e_2
```

Deep learning interpretation:

Embedding coordinates are values along learned basis directions.

The dimensions are useful to the model, but they are not always directly human-interpretable.

</details>

---

### 💻 Code task

Reconstruct a vector from basis vectors.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])

x = 4 * e1 - 3 * e2

print(x)
```

Output:

```python
[ 4. -3.]
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

e1 = torch.tensor([1.0, 0.0])
e2 = torch.tensor([0.0, 1.0])

x = 4 * e1 - 3 * e2

print(x)
```

Output:

```python
tensor([ 4., -3.])
```

</details>

---

### ⚠️ Common mistake

A basis dimension is not automatically a human concept.

In embeddings, each coordinate may represent a distributed feature.

</details>

---

<details>
<summary><strong>Q8. Dimension</strong> — feature size, latent size, hidden size</summary>

---

### 🧠 Concept

The dimension of a vector is the number of components it has.

In deep learning:

| Vector | Dimension means |
|---|---|
| input vector | number of input features |
| embedding vector | embedding size |
| hidden state | hidden size |
| latent vector | latent dimension |
| query/key/value vector | attention head dimension |

---

### 🧮 Formula

If:

```math
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\cdots \\
x_d
\end{bmatrix}
```

then:

```math
x \in \mathbb{R}^d
```

---

### 🎯 Problem

A transformer hidden tensor has shape:

```text
(batch, sequence_length, d_model) = (2, 5, 8)
```

Answer:

1. How many token vectors are in each sequence?
2. What is the dimension of each token vector?
3. How many total token vectors are in the batch?

---

<details>
<summary><strong>✅ Show answer</strong></summary>

The tensor shape is:

```text
(2, 5, 8)
```

So:

| Quantity | Value |
|---|---:|
| batch size | `2` |
| sequence length | `5` |
| vector dimension | `8` |

Each sequence has:

```math
5
```

token vectors.

Each token vector has:

```math
8
```

dimensions.

The full batch has:

```math
2 \times 5 = 10
```

token vectors.

Deep learning interpretation:

Each token is represented by one `d_model`-dimensional vector.

</details>

---

### 💻 Code task

Create a fake hidden tensor and inspect vector dimensions.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

hidden = np.zeros((2, 5, 8))

batch_size, seq_len, d_model = hidden.shape

print("batch size:", batch_size)
print("sequence length:", seq_len)
print("vector dimension:", d_model)
print("total token vectors:", batch_size * seq_len)
```

Output:

```python
batch size: 2
sequence length: 5
vector dimension: 8
total token vectors: 10
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

hidden = torch.zeros((2, 5, 8))

batch_size, seq_len, d_model = hidden.shape

print("batch size:", batch_size)
print("sequence length:", seq_len)
print("vector dimension:", d_model)
print("total token vectors:", batch_size * seq_len)
```

Output:

```python
batch size: 2
sequence length: 5
vector dimension: 8
total token vectors: 10
```

</details>

---

### ⚠️ Common mistake

Do not confuse number of vectors with vector dimension.

In `(batch, sequence_length, d_model)`, the vector dimension is `d_model`.

</details>

---

<details>
<summary><strong>Q9. Norms</strong> — weight decay, gradient clipping, embedding magnitude</summary>

---

### 🧠 Concept

A norm measures vector size.

In deep learning, norms are used for:

| Use case | Meaning |
|---|---|
| gradient clipping | stop gradients from becoming too large |
| weight decay | discourage overly large weights |
| embedding normalization | control representation scale |
| distance calculation | compare vector differences |

---

### 🧮 Formula

L2 norm:

```math
\|x\|_2 =
\sqrt{x_1^2 + x_2^2 + \cdots + x_d^2}
```

---

### 🎯 Problem

A gradient vector is:

```math
g =
\begin{bmatrix}
3 \\
4
\end{bmatrix}
```

Compute:

```math
\|g\|_2
```

If the clipping threshold is `2.5`, should this gradient be clipped?

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
\|g\|_2 = \sqrt{3^2 + 4^2}
```

```math
\|g\|_2 = \sqrt{9 + 16}
```

```math
\|g\|_2 = 5
```

Since:

```math
5 > 2.5
```

the gradient should be clipped.

Deep learning interpretation:

Gradient clipping limits update size and can prevent unstable training.

</details>

---

### 💻 Code task

Compute the gradient norm.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

g = np.array([3.0, 4.0])
threshold = 2.5

norm = np.linalg.norm(g)
should_clip = norm > threshold

print("norm:", norm)
print("should clip:", should_clip)
```

Output:

```python
norm: 5.0
should clip: True
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

g = torch.tensor([3.0, 4.0])
threshold = 2.5

norm = torch.linalg.vector_norm(g)
should_clip = norm > threshold

print("norm:", norm)
print("should clip:", should_clip.item())
```

Output:

```python
norm: tensor(5.)
should clip: True
```

</details>

---

### ⚠️ Common mistake

Large norm is not always good.

Large gradients can make training unstable.

Large weights can overfit or make the model sensitive.

</details>

---

<details>
<summary><strong>Q10. Unit vectors</strong> — normalized embeddings and direction-only comparison</summary>

---

### 🧠 Concept

A unit vector has norm `1`.

Unit vectors keep direction but remove magnitude.

In deep learning, this matters when comparing embeddings by direction instead of length.

This is common in semantic search, contrastive learning, and recommendation systems.

---

### 🧮 Formula

```math
\hat{x} =
\frac{x}{\|x\|_2}
```

---

### 🎯 Problem

Normalize:

```math
x =
\begin{bmatrix}
3 \\
4
\end{bmatrix}
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

The norm is:

```math
\|x\|_2 = 5
```

Normalize:

```math
\hat{x}
=
\frac{1}{5}
\begin{bmatrix}
3 \\
4
\end{bmatrix}
```

```math
\hat{x}
=
\begin{bmatrix}
0.6 \\
0.8
\end{bmatrix}
```

Check:

```math
\sqrt{0.6^2 + 0.8^2} = 1
```

Deep learning interpretation:

Normalized embeddings can be compared fairly by direction.

Magnitude no longer dominates the comparison.

</details>

---

### 💻 Code task

Normalize a vector.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([3.0, 4.0])

x_unit = x / np.linalg.norm(x)

print(x_unit)
print("norm:", np.linalg.norm(x_unit))
```

Output:

```python
[0.6 0.8]
norm: 1.0
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([3.0, 4.0])

x_unit = x / torch.linalg.vector_norm(x)

print(x_unit)
print("norm:", torch.linalg.vector_norm(x_unit))
```

Output:

```python
tensor([0.6000, 0.8000])
norm: tensor(1.)
```

</details>

---

### ⚠️ Common mistake

Do not normalize a zero vector.

A zero vector has no direction.

</details>

---

<details>
<summary><strong>Q11. Distance between vectors</strong> — retrieval, nearest neighbors, clustering</summary>

---

### 🧠 Concept

Distance measures how far two vectors are.

In machine learning, vector distance is used for:

| Use case | Meaning |
|---|---|
| nearest-neighbor search | find closest examples |
| embedding retrieval | find similar documents or images |
| clustering | group nearby vectors |
| anomaly detection | detect points far from normal examples |

---

### 🧮 Formula

Euclidean distance:

```math
d(x,y) = \|x-y\|_2
```

For two-dimensional vectors:

```math
d(x,y) =
\sqrt{(x_1-y_1)^2 + (x_2-y_2)^2}
```

---

### 🎯 Problem

A query embedding is:

```math
q =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

A candidate embedding is:

```math
c =
\begin{bmatrix}
4 \\
6
\end{bmatrix}
```

Compute:

```math
d(q,c)
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
q-c =
\begin{bmatrix}
1-4 \\
2-6
\end{bmatrix}
=
\begin{bmatrix}
-3 \\
-4
\end{bmatrix}
```

```math
d(q,c) =
\sqrt{(-3)^2 + (-4)^2}
```

```math
d(q,c) =
\sqrt{9+16}
```

```math
d(q,c) = 5
```

Deep learning interpretation:

If these are embeddings, the candidate is distance `5` away from the query under Euclidean distance.

Smaller distance means closer under this metric.

</details>

---

### 💻 Code task

Compute Euclidean distance.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

q = np.array([1.0, 2.0])
c = np.array([4.0, 6.0])

distance = np.linalg.norm(q - c)

print(distance)
```

Output:

```python
5.0
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

q = torch.tensor([1.0, 2.0])
c = torch.tensor([4.0, 6.0])

distance = torch.linalg.vector_norm(q - c)

print(distance)
```

Output:

```python
tensor(5.)
```

</details>

---

### ⚠️ Common mistake

Distance is scale-sensitive.

If one feature has a much larger numeric scale, it can dominate the distance.

</details>

---

<details>
<summary><strong>Q12. Dot product</strong> — linear layers, attention scores, similarity</summary>

---

### 🧠 Concept

The dot product multiplies matching components and sums the results.

In deep learning, dot products appear in:

| Use case | Dot product role |
|---|---|
| neuron | `w · x` |
| linear layer | input-weight interaction |
| attention | `q · k` similarity |
| retrieval | embedding similarity |
| cosine similarity | normalized dot product |

---

### 🧮 Formula

```math
x \cdot w =
x_1w_1 + x_2w_2 + \cdots + x_dw_d
```

---

### 🎯 Problem

In attention, a query vector is:

```math
q =
\begin{bmatrix}
1 \\
2 \\
0
\end{bmatrix}
```

A key vector is:

```math
k =
\begin{bmatrix}
3 \\
-1 \\
2
\end{bmatrix}
```

Compute the unscaled attention score:

```math
q \cdot k
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

```math
q \cdot k =
1(3) + 2(-1) + 0(2)
```

```math
q \cdot k =
3 - 2 + 0
```

```math
q \cdot k = 1
```

Deep learning interpretation:

This score measures how strongly the query matches the key before scaling and softmax.

In transformer attention, this later becomes part of:

```math
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
```

</details>

---

### 💻 Code task

Compute the dot product.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

q = np.array([1.0, 2.0, 0.0])
k = np.array([3.0, -1.0, 2.0])

score = np.dot(q, k)

print(score)
```

Output:

```python
1.0
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

q = torch.tensor([1.0, 2.0, 0.0])
k = torch.tensor([3.0, -1.0, 2.0])

score = torch.dot(q, k)

print(score)
```

Output:

```python
tensor(1.)
```

</details>

---

### ⚠️ Common mistake

Dot product is not just element-wise multiplication.

Element-wise multiplication creates component-wise products.

Dot product sums those products into one number.

</details>

---

<details>
<summary><strong>Q13. Orthogonality</strong> — decorrelation, PCA, independent directions</summary>

---

### 🧠 Concept

Two vectors are orthogonal when their dot product is zero.

Geometrically, they point in perpendicular directions.

In machine learning, orthogonality connects to:

| Use case | Meaning |
|---|---|
| PCA | principal components are orthogonal |
| decorrelation | reduce redundant feature directions |
| representation analysis | separate independent directions |
| orthogonal initialization | preserve signal magnitude more stably |

---

### 🧮 Formula

```math
x \perp y
```

if:

```math
x \cdot y = 0
```

---

### 🎯 Problem

Are these vectors orthogonal?

```math
x =
\begin{bmatrix}
2 \\
1
\end{bmatrix}
```

```math
y =
\begin{bmatrix}
1 \\
-2
\end{bmatrix}
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

Compute the dot product:

```math
x \cdot y = 2(1) + 1(-2)
```

```math
x \cdot y = 2 - 2
```

```math
x \cdot y = 0
```

Therefore:

```math
x \perp y
```

Deep learning interpretation:

These two representation directions are independent under dot-product geometry.

In PCA, orthogonal directions capture different variance directions.

</details>

---

### 💻 Code task

Check orthogonality.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([2.0, 1.0])
y = np.array([1.0, -2.0])

dot = np.dot(x, y)

print("dot:", dot)
print("orthogonal:", np.isclose(dot, 0.0))
```

Output:

```python
dot: 0.0
orthogonal: True
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([2.0, 1.0])
y = torch.tensor([1.0, -2.0])

dot = torch.dot(x, y)

print("dot:", dot)
print("orthogonal:", torch.isclose(dot, torch.tensor(0.0)).item())
```

Output:

```python
dot: tensor(0.)
orthogonal: True
```

</details>

---

### ⚠️ Common mistake

Orthogonal does not mean semantically unrelated in every possible sense.

It means zero dot product in the current vector representation.

</details>

---

<details>
<summary><strong>Q14. Projection</strong> — PCA, least squares, representation direction analysis</summary>

---

### 🧠 Concept

Projection finds the component of one vector that lies along another vector.

In machine learning, projection appears in:

| Use case | Meaning |
|---|---|
| PCA | project data onto principal directions |
| least squares | project target onto column space |
| representation analysis | isolate or remove a direction |
| embedding control | measure how much of a vector points toward a concept direction |

---

### 🧮 Formula

Projection of `x` onto `u`:

```math
\text{proj}_u(x)
=
\frac{x \cdot u}{u \cdot u}u
```

If `u` is a unit vector:

```math
\text{proj}_u(x)
=
(x \cdot u)u
```

---

### 🎯 Problem

Project:

```math
x =
\begin{bmatrix}
3 \\
4
\end{bmatrix}
```

onto:

```math
u =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

---

<details>
<summary><strong>✅ Show answer</strong></summary>

Compute:

```math
x \cdot u = 3(1) + 4(0) = 3
```

Compute:

```math
u \cdot u = 1^2 + 0^2 = 1
```

Therefore:

```math
\text{proj}_u(x)
=
\frac{3}{1}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

```math
\text{proj}_u(x)
=
\begin{bmatrix}
3 \\
0
\end{bmatrix}
```

Deep learning interpretation:

Projection extracts how much of a representation lies along a chosen direction.

For example, PCA projects data onto high-variance directions.

</details>

---

### 💻 Code task

Compute a vector projection.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([3.0, 4.0])
u = np.array([1.0, 0.0])

projection = (np.dot(x, u) / np.dot(u, u)) * u

print(projection)
```

Output:

```python
[3. 0.]
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch

x = torch.tensor([3.0, 4.0])
u = torch.tensor([1.0, 0.0])

projection = (torch.dot(x, u) / torch.dot(u, u)) * u

print(projection)
```

Output:

```python
tensor([3., 0.])
```

</details>

---

### ⚠️ Common mistake

Projection is not element-wise multiplication.

Projection uses dot products to measure direction.

</details>

---

<details>
<summary><strong>Q15. Cosine similarity</strong> — semantic search, contrastive learning, recommendation</summary>

---

### 🧠 Concept

Cosine similarity compares vector directions.

It ignores magnitude and focuses on angle.

In deep learning, cosine similarity is used in:

| Use case | Meaning |
|---|---|
| semantic search | compare query and document embeddings |
| recommendation | compare user and item vectors |
| contrastive learning | pull similar pairs closer |
| metric learning | learn embedding geometry |

---

### 🧮 Formula

```math
\cos(\theta)
=
\frac{x \cdot y}{\|x\|_2\|y\|_2}
```

Range:

```math
-1 \leq \cos(\theta) \leq 1
```

| Value | Meaning |
|---:|---|
| `1` | same direction |
| `0` | orthogonal |
| `-1` | opposite direction |

---

### 🎯 Problem

A query embedding is:

```math
x =
\begin{bmatrix}
1 \\
1
\end{bmatrix}
```

A document embedding is:

```math
y =
\begin{bmatrix}
2 \\
2
\end{bmatrix}
```

Compute cosine similarity.

---

<details>
<summary><strong>✅ Show answer</strong></summary>

Dot product:

```math
x \cdot y = 1(2) + 1(2) = 4
```

Norms:

```math
\|x\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2}
```

```math
\|y\|_2 = \sqrt{2^2 + 2^2} = \sqrt{8} = 2\sqrt{2}
```

Cosine similarity:

```math
\frac{4}{(\sqrt{2})(2\sqrt{2})}
=
\frac{4}{4}
=
1
```

Deep learning interpretation:

The query and document embeddings point in the same direction.

For cosine-based retrieval, they are maximally similar even though `y` has larger magnitude.

</details>

---

### 💻 Code task

Compute cosine similarity.

<details>
<summary><strong>🔢 Show NumPy code</strong></summary>

```python
import numpy as np

x = np.array([1.0, 1.0])
y = np.array([2.0, 2.0])

cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(cosine)
```

Output:

```python
0.9999999999999998
```

</details>

<details>
<summary><strong>🔥 Show PyTorch code</strong></summary>

```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 1.0])
y = torch.tensor([2.0, 2.0])

cosine = F.cosine_similarity(x, y, dim=0)

print(cosine)
```

Output:

```python
tensor(1.0000)
```

</details>

---

### ⚠️ Common mistake

Cosine similarity ignores magnitude.

That is useful for semantic similarity, but sometimes magnitude contains important information.

</details>
