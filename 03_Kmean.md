# 📊 K-Means Clustering — From Scratch 
  
## 🧠 What is K-Means? 
 
**K-Means** is an **unsupervised learning** algorithm used for **clustering** — grouping similar data points together.

It tries to divide a dataset into **K distinct, non-overlapping clusters**.

---

## 🧮 How does K-Means work?

1. **Choose K** (number of clusters)
2. **Initialize** K centroids randomly
3. **Repeat until convergence**:
   - Assign each point to the **nearest centroid** (cluster assignment)
   - Move each centroid to the **mean of its assigned points**

This is called an **Expectation-Maximization (EM)** style algorithm.

---

## 📌 Visual Intuition

### Step 1: Initial random centroids
![Initial Centroids](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/400px-K-means_convergence.gif)

### Step 2: Assign points to nearest centroid

### Step 3: Recalculate centroid positions

Repeat steps 2 and 3 until the centroids no longer move.

---

## 🧾 Example: K=3

Given a 2D dataset:

```
•   •      •       (Cluster 1)
        •   •   •       (Cluster 2)
                      •   •   •   (Cluster 3)
```
K-Means will:
- Randomly place 3 centroids
- Assign each point to the nearest centroid
- Recompute centroids as the average of their assigned points
- Repeat

---

## 📉 Objective: Minimize Inertia (SSE)

The goal is to minimize **inertia**, i.e., the sum of squared distances between each point and its centroid:
![image](https://github.com/user-attachments/assets/f94c25fb-8f12-4a79-8f22-d989dcde2063)

\[
J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
\]

Where:
- \( K \): number of clusters
- \( \mu_i \): centroid of cluster \( i \)
- \( C_i \): points in cluster \( i \)

---

## 🧮 K-Means from Scratch (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x1 = np.random.normal(0, 1, (50, 2))
x2 = np.random.normal(5, 1, (50, 2))
x3 = np.random.normal(10, 1, (50, 2))
X = np.vstack((x1, x2, x3))

# K-means implementation
K = 3
centroids = X[np.random.choice(range(len(X)), K, replace=False)]

for _ in range(100):
    # Assign clusters
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
    
    # Stop if converged
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Plot result
for i in range(K):
    plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f"Cluster {i+1}")
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
```

---

## 🔍 Choosing K: The Elbow Method

Plot inertia (within-cluster sum of squares) for different K values:

```python
inertias = []
for k in range(1, 10):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(100):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
    inertia = sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1).sum() for i in range(k))
    inertias.append(inertia)

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

Look for the 'elbow' point where inertia stops decreasing sharply.

---

## ✅ Summary

- K-Means clusters data into K groups based on distance
- Uses iterative optimization: assign → update → repeat
- Final result depends on **initial centroid positions** (can use K-Means++)
- Use the **Elbow Method** to find the best K

---

Next: DBSCAN or Hierarchical Clustering?

