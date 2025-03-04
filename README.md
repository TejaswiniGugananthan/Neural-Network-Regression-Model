# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![Screenshot 2025-03-03 114334](https://github.com/user-attachments/assets/72cd7f3c-501f-4064-b858-5872c0cdefd4)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
# Name: G.TEJASWINI
# Register Number: 212222230157

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        class NeuralNet(nn.Module):
          self.fc1 = nn. Linear (1, 3)
          self.fc2 = nn. Linear (3, 2)
          self.fc3 = nn. Linear (2, 1)
          self.relu = nn. ReLU()
          self.history = {'loss': []}
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self. fc3(x)
    return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (ai_brain. parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss. backward()
    optimizer.step()
    ai_brain. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

![image](https://github.com/user-attachments/assets/b6905018-fc86-435a-8e52-27f12f55a82f)


## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
