from nn import MLP
from engine import Value
import numpy as np

model = MLP(n_inputs=3, n_outs=[4, 4, 1])

x = np.array([[-2.0, 1.2, 2.0],
              [1.8, 1.0, 0.0],
              [-2.0, 1.0, 1.2],
              [0.5, -1.2, 2.0]])

y = np.array([1, -1, 1, -1])

epochs = 500
def train():
    for epoch in range(epochs):
        # Forward pass
        ypred = model(x)
        ypred = ypred.flatten()

        # Calculate the loss
        loss = Value(0.0)
        n_samples = len(y)
        for ygt, yout in zip(y, ypred):
            diff = yout - ygt
            loss = loss + (diff ** 2)
        loss = loss * (1.0 / n_samples)

        # Backward pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # Update
        for p in model.parameters():
            p.data += -0.01 * p.grad

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data}")

train()

ypred_final = model(x)
ypred_final = ypred_final.flatten()

print(ypred_final)