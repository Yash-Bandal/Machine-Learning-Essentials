# 📘 Linear Regression (From Scratch)

# Note Dataframe -> StandardScalar --> Array --> Model .Model takes in array

## 📍 What is Regression?  
  
Regression is a way to **predict continuous values** using data.

Definition - In machine learning, regression is a supervised learning technique used to predict a continuous numerical value based on one or more input features. It models the relationship between independent variables (features) and a dependent variable (target), allowing for predictions on unseen data.
🔍 Example:  
- Predict someone's **weight** from their **height**  
- Predict a house's **price** from its **size**

---

## 📈 Let's Start with a Line

A **line** in 2D looks like:
```
y = mx + c
```

In machine learning, we write it as:
```
ypred = wx + b
```

- `x`: input (independent variable)  
- `y`: output (dependent variable)  
- `w`: slope of the line (also called weight)  
- `b`: intercept (also called bias)  
- `ypred`: predicted output

---

## 📉 Real Data is Noisy

In real life, data is **not perfectly on a line**.

Example: Plotting `height` vs `weight` for 5 people.

Imagine a scatter plot like this:

```
     |
 70  |      •
 65  |    •   •
 60  |  •
 55  |        •
     |----------------
       160 170 180 190
            Height
```

You can draw a line **that best fits** all the points.

---

## 📏 What are Residuals?

**Residual** = difference between the actual value and the predicted value.

```
Residual = y - ypred
```

- If the line is a perfect fit → residuals = 0  
- The smaller the residuals, the better the fit.

---

## 🎯 Goal of Linear Regression

Find the line that makes all residuals **as small as possible**.

We use a method called **Least Squares**.

---

## 🛆 Least Squares Method

We square the residuals (to make all values positive) and add them up:

```
Total Error = sum (yi - ypred i)^2
```

This is called the **Sum of Squared Errors (SSE)**.

The best line is the one that **minimizes** this total squared error.

---

## 📉 Visual: Line + Residuals

Here's an example plot:

![Linear Regression with Residuals](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

- Blue points = actual data  
- Red line = best fit line  
- Vertical lines = residuals (errors)

---

## 🧠 Summary of Process

1. **Plot the data**
2. **Fit a line**: \(\hat{y} = wx + b\)
3. **Calculate residuals**: \(y - \hat{y}\)
4. **Use Least Squares** to find the best `w` and `b`
5. Done ✅

---

## 📊 Real Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, label="Actual Points")
plt.plot(X, y_pred, color='red', label="Best Fit Line")
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```

---

## ✅ Key Takeaways

- Linear regression fits a **straight line** through data.
- We use the **least squares method** to minimize errors.
- The model finds the best slope (`w`) and intercept (`b`) to predict `y`.

---

## ➕ Want to Learn Next?

- Multiple Linear Regression (more than one input)  
- Gradient Descent (how we optimize w and b)  
- Cost function details  
- Polynomial Regression (when data isn't linear)

