# PyTorch Geometric Report

## 1. Changes made in the lab notebooks

### L_03_1 - another graph
- I replaced the original 3-node toy graph with a 5-node campus map graph.
- Nodes represent the dormitory, library, cafeteria, laboratory, and sports hall.
- Node features were expanded to two values per node: location importance and average daily traffic.
- The notebook now shows the same graph in two construction styles and prints the main graph properties.

Observed result:
- Graph shape: `Data(x=[5, 2], edge_index=[2, 12])`
- Number of nodes: `5`
- Number of edges: `12`
- Self-loops: `False`
- Isolated nodes: `False`

### L_03_2 - another dataset
- I replaced the `ENZYMES` dataset with `MUTAG`.
- I also changed the storage path to a local relative folder (`data/TUDataset`) so the notebook works cleanly in this workspace.
- The short documentation inside the notebook explains why `MUTAG` is an appropriate alternative benchmark.

Observed result:
- Dataset summary: `MUTAG(188)`
- Number of graphs: `188`
- Number of classes: `2`
- Number of node features: `7`
- First sample graph: `Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])`
- Undirected sample graph: `True`

### L_03_3 - improved accuracy
- I kept the notebook close to the original GCN example, but improved the pipeline in four ways:
- Input features are normalized with `NormalizeFeatures()`.
- Random seeds are fixed for reproducibility.
- The hidden layer width was increased from `16` to `32`.
- The training loop now tracks validation accuracy and restores the checkpoint with the best validation score.

Observed result:
- Original baseline accuracy before modification: `0.8060`
- Best epoch after modification: `210`
- Best validation accuracy: `0.8040`
- Final test accuracy with validation-selected checkpoint: `0.8120`

## 2. Official PyG Colab notebook 1 report

Local copy used:
- `Laborator/colab/pyg_colab_1.ipynb`

What was adapted locally:
- Removed Colab-specific package installation commands because `torch` and `torch-geometric` are already installed in the workspace environment.
- Removed Colab-only output resizing code.
- Reduced visualization frequency during training so the notebook remains readable locally.

Main observations:
- Dataset: `KarateClub()`
- Number of graphs: `1`
- Number of features: `34`
- Number of classes: `4`
- Graph statistics: `34` nodes, `156` edges, average degree `4.59`
- The untrained GCN already produced a partially structured 2D embedding.
- During training, the loss decreased from about `1.4324` at epoch `0` to about `0.0246` at epoch `400`.
- The trained embeddings became clearly separated by community, which matches the educational goal of the notebook.

## 3. Official PyG Colab notebook 2 report

Local copy used:
- `Laborator/colab/pyg_colab_2.ipynb`

What was adapted locally:
- Removed Colab-specific package installation commands.
- Removed Colab-only output resizing code.
- Added a separate solution cell for Exercise 1.

Main observations from the original notebook flow:
- Dataset: `Cora()`
- Number of nodes: `2708`
- Number of edges: `10556`
- Number of training nodes: `140`
- MLP test accuracy: `0.5740`
- GCN test accuracy: `0.8110`

Interpretation:
- The MLP uses only node features and performs much worse.
- The GCN uses both node features and graph connectivity, which leads to a large improvement.
- The final trained t-SNE visualization shows much clearer class clustering than the untrained embedding.

## 4. Exercise 1 report for Colab notebook 2

Task solved:
- I modified the training logic to keep the model checkpoint with the highest validation accuracy using `data.val_mask`.
- After training, the notebook restores that checkpoint and reports the corresponding test accuracy.

Observed result:
- Best epoch: `172`
- Best validation accuracy: `0.7980`
- Test accuracy selected by validation: `0.8150`

Comparison:
- Improvement over the notebook's plain GCN run: `0.8150 - 0.8110 = 0.0040`
- Improvement over the MLP baseline: `0.8150 - 0.5740 = 0.2410`

Conclusion:
- Even a simple validation-based checkpoint selection improves the final result.
- The main benefit is not a radical architecture change, but selecting the most generalizable model instead of the last trained epoch.
