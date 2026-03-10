# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a neural network regression model using a custom dataset containing one numeric input and one numeric output. The dataset is created in Google Sheets, and the model is trained to learn the relationship between the input and output values. During training, the model minimizes prediction error, and a training loss vs iteration plot is generated to visualize how the model improves over time.

## Neural Network Model

<img width="730" height="375" alt="image" src="https://github.com/user-attachments/assets/9b92a9ee-8282-475e-b394-a155528c25ec" />


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
### Name:Thirunavukkarasu meenakshisundaram
### Register Number: 212224220117
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x
        



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')




```
## Dataset Information



<img width="347" height="950" alt="image" src="https://github.com/user-attachments/assets/04b8a9cd-d7c3-4a7a-9a0b-5936c857f884" />



## OUTPUT

### Training Loss Vs Iteration Plot


<img width="688" height="475" alt="image" src="https://github.com/user-attachments/assets/54b60a38-93a2-4483-a067-11c3757e3119" />


### New Sample Data Prediction



<img width="818" height="174" alt="image" src="https://github.com/user-attachments/assets/ffab6408-8d96-43c6-bb10-37e430312b93" />

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.


