# HEGN: Hierarchical Equivariant Graph Neural Network for 9DoF Point Cloud Registration

This repository is the non official implementation of the paper:

> **Hierarchical Equivariant Graph Neural Network for 9DoF Point Cloud Registration*

## 🏗 Setup
1. Clone and set up the HEGEN repository
```bash
git clone 
cd HEGN

# set up virtual environment.
conda env create -f environment.yaml
conda activate hegn

# set up the HEGN python package
pip install -e .
```

## 🖼️ Prepare dataset

### Synthetic ModelNet40 dataset
HEGN is evaluated on the synthetic ModelNet40 dataset. to download the dataset, run the following command:
```bash
python dataloader/dataloader.py
```

### Real-world ScanObjectNN dataset
HEGN is evaluated on the real-world ScanObjectNN dataset. to download the dataset, run the following command:
```bash

```

## :rocket: Run 

### train HEGN on ModelNet40 dataset
```bash
python scripts/train.py
```

### Evaluate HEGN on ModelNet40 dataset
```bash
python scripts/evaluate.py
```

### train HEGN on ScanObjectNN dataset
```bash

```

### Evaluate HEGN on ScanObjectNN dataset
```bash

```


## 📔 Abstract

Given its wide application in robotics, point cloud registration is a widely researched topic. Conventional methods aim to find a rotation and translation that align two point clouds in 6 degrees of freedom (DoF). However, certain tasks in robotics, such as category-level pose estimation, involve non- uniformly scaled point clouds, requiring a 9DoF transform for accurate alignment. We propose HEGN, a novel equivariant graph neural network for 9DoF point cloud registration. HEGN utilizes equivariance to rotation, translation, and scaling to estimate the transformation without relying on point corre- spondences. Based on graph representations for both point clouds, we extract equivariant node features aggregated in their local, cross-, and global context. In addition, we introduce a novel node pooling mechanism that leverages the cross-context importance of nodes to pool the graph representation. By repeating the feature extraction and node pooling, we obtain a graph hierarchy. Finally, we determine rotation and translation by aligning equivariant features aggregated over the graph hierarchy. To estimate scaling, we leverage scale information in the vector norm of the equivariant features. We evaluate the effectiveness of HEGN through experiments with the synthetic ModelNet40 dataset and the real-world ScanObjectNN dataset. The results show the superior performance of HEGN in 9DoF point cloud registration and its competitive performance in conventional 6DoF point cloud registration.


## 👩‍⚖️  License

For academic usage, the code is released under the [MIT](https://opensource.org/licenses/MIT) license.
For any commercial purpose, please contact the authors. 
