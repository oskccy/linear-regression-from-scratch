# Linear Regression with Gradient Descent + MinMax Normalization
Linear Regression using batch gradient descent: FROM SCRATCH!
> By: [Oscar Sharaz Spencer](https://www.linkedin.com/in/oscar-sharaz/)

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
