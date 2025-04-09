# 🧠 Support Vector Machine (SVM) — From Scratch

## 📌 What is SVM?

**Support Vector Machine (SVM)** is a supervised learning algorithm used for **classification** and sometimes **regression**.

Its main goal is to **find the best decision boundary (hyperplane)** that separates classes.

---

## ✂️ Linear Classification Example

Imagine 2 classes of points:
- Red = Class 1
- Blue = Class 2

We want to draw a line that **separates them** as clearly as possible.

```
• • • •        (Red)
        ------ Decision boundary ------
            • • • •   (Blue)
```

This line is called the **Hyperplane**.

### 📏 But which line is the best?
SVM chooses the one with the **maximum margin**.

Margin = distance between the line and the nearest points from each class.

---

## 📐 Support Vectors

- The data points **closest to the hyperplane** are called **Support Vectors**.
- They are the **most important** because they **define the margin**.

![SVM Margin and Support Vectors](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png)

- Blue line = decision boundary (hyperplane)
- Dashed lines = margins
- Dots on margin = support vectors

---

## ⚙️ SVM Optimization Goal

SVM tries to **maximize the margin** (distance between classes).

### Math: Objective Function
If we write the hyperplane as:
\[ w \cdot x + b = 0 \]

Then the goal is to:
\[ \min \frac{1}{2} ||w||^2 \quad \text{subject to } y_i (w \cdot x_i + b) \geq 1 \]

This is a **convex optimization problem**.

---

## ❗ What if data is not linearly separable?

Sometimes we can't draw a straight line to separate classes:

![Non-linearly separable](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/512px-Kernel_Machine.svg.png)

We solve this with:
- **Soft Margin**: Allows some misclassification
- **Kernel Trick**: Transform data to a higher dimension where it’s linearly separable

[Source: Wikipedia – Kernel Machine](https://en.wikipedia.org/wiki/File:Kernel_Machine.svg)

---

## 🧱 Hard Margin vs Soft Margin

### 🔒 Hard Margin SVM
- No points allowed inside the margin
- Works **only** when data is **perfectly separable**
- Very sensitive to noise (outliers break it)

![image](https://miro.medium.com/v2/resize:fit:1400/1*RgFWpCEG5AvnmGF5ESy1Tg.png)





### 🧻 Soft Margin SVM
- Allows **some** points to be inside the margin or misclassified
- More flexible and **works well on real-world data**
- Controlled by a parameter `C`



-
### The C Parameter
- `C` controls **trade-off between margin size and classification error**
- Small `C`: Larger margin, more tolerance for misclassification
- Large `C`: Smaller margin, less tolerance for error (stricter)

---

## 🧙‍♂️ Kernel Trick

**Kernel** = mathematical function that transforms the input space.

### Common Kernels:
- **Linear Kernel**: No transformation (use when data is already linear)
- **Polynomial Kernel**: Adds curved boundaries
- **RBF (Radial Basis Function)** / **Gaussian**: Best for circular / complex boundaries

### Example (RBF Kernel):

![SVM RBF Kernel](https://scikit-learn.org/stable/_images/sphx_glr_plot_rbf_parameters_001.png)

Red and blue dots are perfectly separated after using a non-linear kernel.

---

## 🔧 SVM in Scikit-Learn (with `SVC`)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# Train-test split
X_train, X_test, y_train
```
---


## 🧠 SVC Kernels

| Kernel      | Use Case                              | Description                                                  |
|-------------|----------------------------------------|--------------------------------------------------------------|
| `linear`    | Data is linearly separable             | Separates data using a straight line or hyperplane.          |
| `rbf`       | Non-linear, default & most used        | Maps input space to higher dimension using radial function.  |
| `poly`      | Polynomial relationships in data       | Uses a polynomial kernel with a tunable degree.              |
| `sigmoid`   | Mimics neural network behavior         | Uses the sigmoid function (less commonly used).              |

### ✅ When to Use Which:
- Use `linear` when data is clearly linearly separable.
- Use `rbf` when you're unsure or data has complex patterns.
- Use `poly` if you suspect curved boundaries.
- Use `sigmoid` mainly for experimentation.

---

## 🔧 Hyperparameters in SVC

### 1. `C` (Regularization parameter)
- Controls the trade-off between a smooth decision boundary and classifying training points correctly.
- **Small C**: Wider margin, more tolerant of misclassification.
- **Large C**: Narrow margin, aims for perfect classification (can overfit).

### 2. `kernel`
- Defines how to transform the data for finding the hyperplane.

### 3. `gamma`
- Used with `rbf`, `poly`, and `sigmoid`.
- **High gamma**: More curved, complex boundaries (can overfit).
- **Low gamma**: Smoother boundaries.

### 4. `degree`
- Only for `poly` kernel. Controls the degree of the polynomial function.

---

## 📈 Visualization Examples

### Linear Kernel (2D example)
```
Class A: o o o
Class B: x x x

Best separating hyperplane:

    o   o    o
    |--------| ← hyperplane
    x   x    x
```

### RBF Kernel (non-linear)
```
Class A: scattered inside a circle
Class B: surrounding the circle

RBF transforms the circle into a linearly separable shape in higher dimension.
```

---

- Experiment with kernels on datasets like `Iris`, `Digits`, `Wine`.

---


## 🧾 Summary

- SVM tries to **find the best separating hyperplane**
- **Support vectors** are the key boundary points
- SVM can work with **non-linear data** using **kernels**
- Use `SVC` in scikit-learn for implementation

---

## 🧠 Next Topics

- Soft Margin vs Hard Margin
- C Parameter (controls trade-off)
- Gamma in RBF Kernel
- Multiclass SVM (One-vs-One, One-vs-Rest)
- Visualization of decision boundaries in 2D





