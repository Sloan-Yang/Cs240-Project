# CS240 Project: Accelerating Metric Learning with Divide-and-Conquer

## **Introduction**
This project focuses on accelerating metric learning by employing a divide-and-conquer approach. Metric learning is a fundamental technique in machine learning, often used for tasks like image retrieval, clustering, and classification. By dividing the dataset and model into manageable subproblems, training can be parallelized and optimized for efficiency. After training, the sub-models are merged and fine-tuned to ensure global consistency.

---

## **Key Features**
1. **Data and Model Partitioning**:
   - The dataset is divided into multiple clusters using clustering algorithms (e.g., K-Means or Faiss).
   - The embedding layer of the model is split into corresponding subspaces.

2. **Parallel Training**:
   - Each sub-model trains independently on its assigned data cluster.
   - This step ensures that each sub-model optimizes for a specific part of the embedding space.

3. **Merging and Fine-Tuning**:
   - After the independent training phase, all sub-models are merged into a single unified model.
   - A fine-tuning step is performed on the full dataset to ensure global consistency and optimal performance.

---

## **Implementation Details**

### **Algorithm Implementation**
The main implementation of the algorithm can be found in the following files:
- `train.py`: Handles the overall training pipeline, including dataset preparation, model training, and fine-tuning.
- `lib/model.py`: Defines the model architecture and its components, including embedding layers and backbone.
- `lib/clustering.py`: Implements the clustering algorithms used for dataset partitioning, such as K-Means and Faiss.

### **Dataset Partitioning**
- Data is clustered into K groups using algorithms like:
  - K-Means (via sklearn)
  - Faiss for GPU-accelerated clustering
- Each cluster corresponds to a specific subspace of the embedding space.

### **Model Partitioning**
- The embedding layer of the model is split into K subspaces, each responsible for a specific data cluster.
- For example, if the embedding size is d and the number of clusters is K, each sub-model learns a subspace of size d/K.

### **Training Procedure**
1. **Independent Training**:
   - Each sub-model trains on its respective cluster using a custom loss function (e.g., Triplet Loss or Margin Loss).
   - Gradients are only applied to the parameters of the subspace and the shared backbone.

2. **Reclustering** (Optional):
   - The dataset is dynamically reclustered after a fixed number of epochs to adapt to the evolving embedding space.

3. **Merging and Fine-Tuning**:
   - All subspaces are merged into a single embedding layer.
   - The model is fine-tuned on the full dataset for global optimization.


---

## **Results and Evaluation**
### **Datasets**
The method has been tested on datasets like:
- In-Shop Clothes
- Fashion-MNIST

### **Metrics**
- Recall@k
- Normalized Mutual Information (NMI)

### **Performance**
- Significant speed-up in training due to parallelization.
- Improved convergence during the fine-tuning phase.

---

## **How to Use**

### **Setup**
1. Clone the repository:
   ```bash
   git clone git@github.com:Sloan-Yang/Cs240-Project.git
   cd cs240_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Run Training**
```bash
python experiment.py --dataset=inshop \
--dir=test --exp=0 --random-seed=0 --nb-clusters=8 --nb-epochs=200 \
--sz-batch=80 --backend=faiss-gpu  --embedding-lr=1e-5 --embedding-wd=1e-4 \
--backbone-lr=1e-5 --backbone-wd=1e-4 --finetune-epoch=190
```
- Customize the configuration file for different datasets or clustering settings.



---

## **Future Work**
- Explore other clustering methods for better data partitioning.
- Integrate with large-scale datasets for further validation.
- Investigate methods to improve the merging and fine-tuning process.

---



---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.
