# Emerging Directions in Deep Learning - Lab 5
Student: Pascu Ionut Alexandru, group 507

---

## Overview

This lab is based on two tutorials from the **Design of Graph Neural Networks** section of PyTorch Geometric:

1. **Creating Message Passing Networks**
2. **Heterogeneous Graph Learning**

The goal was not just to run some PyG code, but to understand what these tutorials are trying to teach.

The first tutorial explains how many graph neural network layers can be seen as instances of the same general **message passing** idea. The second tutorial shows what changes when a graph is no longer made of only one kind of node and one kind of edge, but instead contains multiple entity types and relation types.

For this lab I created two notebooks:
- `L_05_1.ipynb` for message passing and the end-of-tutorial exercises
- `L_05_2.ipynb` for heterogeneous graph learning and runnable typed-graph examples

---

## Tutorial 1 - Creating Message Passing Networks

### What the tutorial is about

This tutorial explains the core abstraction behind many GNN layers: **each node receives information from its neighbors, aggregates it, and updates its own representation**.

In PyG, this general idea is written as:

$$x_i^{(k)} = \gamma^{(k)}\left(x_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(k)}(x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i})\right)$$

In simpler words:
- $\phi$ builds the message that a neighbor sends
- $\bigoplus$ aggregates all incoming messages using something like sum, mean or max
- $\gamma$ updates the node after aggregation

This is exactly what the `MessagePassing` class in PyG automates.

### Key elements

The tutorial then shows that, once this pattern is clear, well-known layers such as **GCN** and **EdgeConv** become much easier to understand.

#### GCN

The GCN layer is based on the idea that each node should receive a normalized sum of neighbor features. Self-loops are added so that a node also keeps part of its own information.

A very important part here is the degree-based normalization. Nodes with many neighbors should not dominate just because they are highly connected.

#### EdgeConv

EdgeConv is more geometric. Instead of only summing transformed neighbor features, it looks at the pair:
- the central node feature $x_i$
- the relative difference $x_j - x_i$

This makes the layer especially useful for point clouds and local geometric structure, because it encodes not only who the neighbor is, but also how that neighbor differs from the current node.

### What I ran

In `L_05_1.ipynb`, I:
- implemented a custom `GCNConv` layer from the tutorial logic
- implemented a custom `EdgeConv` layer
- ran both on small example graphs / point sets
- solved the exercises from the end of the tutorial on the exact toy `Data` object proposed there

### Exercise results

For the tutorial graph with three nodes and edges between neighbors, I obtained the following concrete values:

- `row = [0, 1, 1, 2, 0, 1, 2]`
- `col = [1, 0, 2, 1, 0, 1, 2]`
- `degree(col) = [2.0, 3.0, 2.0]`
- `deg_inv_sqrt[row] = [0.7071, 0.5774, 0.5774, 0.7071, 0.7071, 0.5774, 0.7071]`
- `deg_inv_sqrt[col] = [0.5774, 0.7071, 0.7071, 0.5774, 0.7071, 0.5774, 0.7071]`
- `norm = [0.4082, 0.4082, 0.4082, 0.4082, 0.5, 0.3333, 0.5]`
- `x_j` with identity linear map = `[-1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0]`

These values make the theory much clearer:
- `row` contains the **source nodes** of the edges
- `col` contains the **target nodes**
- `degree(col, ...)` is used because GCN normalizes according to the nodes that receive messages
- `x_j` is the source feature lifted to edge level

For the `EdgeConv` exercise, the concatenated edge-level tensor had shape **[4, 2]**, with rows:

- `[0.0, -1.0]`
- `[-1.0, 1.0]`
- `[1.0, -1.0]`
- `[0.0, 1.0]`

This confirms the meaning of `torch.cat([x_i, x_j - x_i], dim=1)`: each row corresponds to one edge, and the two columns store the central-node value and the relative difference.

### Main takeaway from tutorial 1

This tutorial is useful because it removes some of the mystery from PyG. Instead of treating layers such as GCN and EdgeConv as black boxes, we see that they are concrete choices of:
- what message is sent,
- how messages are aggregated,
- and how the node is updated afterwards.

---

## Tutorial 2 - Heterogeneous Graph Learning

### What the tutorial is about

The second tutorial moves from ordinary graphs to **heterogeneous graphs**.

A homogeneous graph has one node type and one edge type. A heterogeneous graph can contain multiple types of entities and multiple types of relations.

Typical real examples are recommendation systems, citation networks with authors and papers, or academic graphs with authors, papers, institutions and topics.

### Why heterogeneous graphs are different

In a homogeneous graph, one feature matrix and one edge index are enough.

In a heterogeneous graph, this is no longer true because:
- different node types may have different meanings
- different edge types represent different relations
- each type may need its own tensors and possibly its own message-passing rule

This is why PyG uses **`HeteroData`**.

### Key elements

#### HeteroData

The tutorial shows that a heterogeneous graph is stored by type:
- node features are grouped by node type
- edges are grouped by edge triplets of the form `(source_type, relation_type, destination_type)`

This is the core mental model for working with hetero graphs in PyG.

#### Transforms

The tutorial also highlights several useful transforms:
- `ToUndirected()` adds reverse relations where needed
- `AddSelfLoops()` adds self-information when appropriate
- `NormalizeFeatures()` rescales features

In heterogeneous graphs, these transforms still work, but they operate in a typed way.

#### Building heterogeneous GNNs

The tutorial presents three main approaches:

1. **`to_hetero()`**
   Start from a homogeneous GNN and automatically convert it into a heterogeneous one.

2. **`HeteroConv`**
   Manually specify a different convolution operator for each relation type.

3. **Dedicated heterogeneous operators**
   Use layers built specifically for hetero graphs, such as HGT-style operators.

### What I ran

Since the default MAG example from the tutorial is large, I used a **small synthetic heterogeneous graph** so that the ideas remain clear and the code stays runnable locally.

My graph used three node types:
- `paper`
- `author`
- `institution`

and three main edge types:
- `author -> writes -> paper`
- `author -> affiliated_with -> institution`
- `paper -> cites -> paper`

### Results from the code

The notebook confirmed the following metadata:

- node types: `paper`, `author`, `institution`
- edge types before transforms:
  - `('author', 'writes', 'paper')`
  - `('author', 'affiliated_with', 'institution')`
  - `('paper', 'cites', 'paper')`

After applying the transforms, the graph contained these edge types:

- `('author', 'writes', 'paper')`
- `('author', 'affiliated_with', 'institution')`
- `('paper', 'cites', 'paper')`
- `('paper', 'rev_writes', 'author')`
- `('institution', 'rev_affiliated_with', 'author')`

This is exactly what we expect from `ToUndirected()`: reverse relation types are added automatically.

For the `to_hetero()` example:
- the model produced outputs for all node types
- output shapes were:
  - `paper`: `[4, 2]`
  - `author`: `[3, 2]`
  - `institution`: `[2, 2]`

I then trained the converted model on the toy `paper` labels, and it reached:
- final predictions: `[0, 0, 1, 1]`
- final accuracy: **1.0** on the tiny graph

This result is not meant as a benchmark. The graph is extremely small, so the model simply overfits it. The point is that the heterogeneous training flow works correctly, and `to_hetero()` produces the expected typed outputs.

For the `HeteroConv` example:
- output shapes were again:
  - `paper`: `[4, 2]`
  - `author`: `[3, 2]`
  - `institution`: `[2, 2]`

This shows that we can also define relation-specific convolutions manually and still obtain valid outputs for each node type.

### Main takeaway from tutorial 2

The most important lesson here is that heterogeneous graphs require **typed message passing**. A single generic adjacency structure is no longer enough. PyG handles this cleanly by grouping data by type and letting us choose how much control we want:
- automatic conversion with `to_hetero()` if we want convenience
- `HeteroConv` if we want relation-specific design
- dedicated hetero operators if we want more specialized models

---

## Final conclusion

These two tutorials fit together very well.

The first tutorial explains the **core mechanism** behind graph neural networks: message passing.
The second tutorial shows how that same mechanism must be adapted when the graph becomes more realistic and contains **multiple types of nodes and relations**.

So, in a way, the second tutorial is a natural extension of the first one:
- first we learn how graph layers work in general
- then we learn how to scale that idea to typed, heterogeneous graphs

---

## Screenshot placeholders

- Add here one screenshot from `L_05_1.ipynb` showing the solved exercise outputs.
- Add here one screenshot from `L_05_2.ipynb` showing the `HeteroData` summary and/or the final `to_hetero()` results.

---

## References

- Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop on Representation Learning on Graphs and Manifolds.
- Kipf, T.N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., & Solomon, J.M. (2019). *Dynamic Graph CNN for Learning on Point Clouds*. ACM TOG.
- Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*. NeurIPS.
