# Linear Regression with Gradient Descent + MinMax Normalization
Linear Regression using batch gradient descent on the [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset), predicting the relation between total house area and house price: from scratch! Rendered in matplotlib.
> By: [Oscar Sharaz Spencer](https://www.linkedin.com/in/oscar-sharaz/)

## Index
1. [Introduction](#introduction)
2. [Min-Max Normalization](#min-max-normalization)
    - [Normalizing Features](#normalizing-features)
3. [Denormalizing Parameters](#denormalizing-parameters)
4. [Gradient Descent](#gradient-descent)
    - [Error Term](#error-term)
    - [Gradient Descent Update Rules](#gradient-descent-update-rules)
5. [Running the Predictions](#running-the-predictions)
    - [main.py](#mainpy)
    - [Installation and Execution](#installation-and-execution)

## Introduction
This repository contains an implementation of linear regression using batch gradient descent, developed from scratch. The `main.py` file is the main entry point for running the predictions. To get started, ensure you have Python 3 installed on your system and follow the installation and execution instructions below:


## Min-Max Normalization

### Normalizing Features
Given a dataset with features $\( x \)$ and $\( y \)$, min-max normalization scales each feature to the range $[0, 1]$.

**Normalization Formula:**
$\[ x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \]$
$\[ y_{\text{norm}} = \frac{y - y_{\min}}{y_{\max} - y_{\min}} \]$

Where:
- $\( x_{\min} \)$ and $\( x_{\max} \)$ are the minimum and maximum values of the feature $\( x \)$.
- $\( y_{\min} \) and \( y_{\max} \)$ are the minimum and maximum values of the feature $\( y \)$.
  
## Denormalizing Parameters
After training the model, the parameters $\( m \)$ and $\( b \)$ need to be scaled back to the original range of the data.

**Denormalization Formula:**
$\[ m_{\text{original}} = m_{\text{normalized}} \times \frac{y_{\max} - y_{\min}}{x_{\max} - x_{\min}} \]$
$\[ b_{\text{original}} = b_{\text{normalized}} \times (y_{\max} - y_{\min}) + y_{\min} - m_{\text{original}} \times x_{\min} \]$

## Gradient Descent

### Error Term
The error term is the difference between the predicted value and the actual value.

**Error Formula:**
$\[ \text{error} = y - \hat{y} \]$
$\[ \hat{y} = m \cdot x + b \]$

### Gradient Descent Update Rules
Gradient descent updates the parameters $\( m \)$ and $\( b \)$ to minimize the error term.

**Gradients:**
$\[ \frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i \cdot (y_i - (m \cdot x_i + b)) \]$
$\[ \frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (m \cdot x_i + b)) \]$

Where $\( J \)$ is the cost function.

**Update Rules:**
$\[ m = m - \alpha \cdot \frac{\partial J}{\partial m} \]$
$\[ b = b - \alpha \cdot \frac{\partial J}{\partial b} \]$

Where:
- $\( \alpha \)$ is the learning rate.
- $\( n \)$ is the number of data points.

## Running the Predictions

### main.py
The `main.py` file is the main entry point for running the linear regression predictions. It initializes the dataset, normalizes the features, runs the gradient descent algorithm, and outputs the predicted results.

### Installation and Execution
To run the predictions, follow these steps:

1. Ensure you have Python 3 installed on your system.
2. Clone this repository and navigate to the project directory.
3. Run the following commands to install any necessary dependencies and execute the script:

```bash
# Clone the repository
git clone https://github.com/oskccy/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# Install dependencies
pip install -r requirements.txt

# Run the predictions
python3 main.py
