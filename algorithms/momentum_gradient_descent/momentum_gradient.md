# Momentum Gradient Descent

## Overview
This module implements **Momentum Gradient Descent**, an optimization algorithm that accelerates gradient descent by adding a fraction of the previous update to the current one. It helps smooth out oscillations and speeds up convergence.

## Algorithm
The update rule is:
- \( v_t = \gamma v_{t-1} + \eta \nabla J(w) \)
- \( w = w - v_t \)

Where:
- \( \gamma \) = momentum coefficient (typically 0.9)
- \( \eta \) = learning rate
- \( v_t \) = velocity term
- \( \nabla J(w) \) = gradient of the loss function

## Code
```python
def momentum_gradient_descent():
    w, b, eta = init_w, init_b, 1.0
    prev_dw, prev_db, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X_train, y_train):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        v_w = gamma * prev_dw + eta * dw
        v_b = gamma * prev_db + eta * db
        w -= v_w
        b -= v_b
        prev_dw, prev_db = v_w, v_b
```
## Uses

import and call momentum_gradient_descent() only after defining:
- init_w,init_b
- X_train,Y_train
- gradient functions grad_w,grad_b
- max_epochs

## Notes
We can adjust learning rate and monmentum (gamma).
This algorithm works best for convex optimization problems.
