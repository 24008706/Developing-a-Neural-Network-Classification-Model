# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the customer dataset and preprocess it by handling missing values and encoding categorical features.

### STEP 2: 
Split the dataset into training and testing sets to evaluate model performance.


### STEP 3: 

Define a neural network architecture with fully connected layers and ReLU activation functions.

### STEP 4: 
Select an appropriate loss function (CrossEntropyLoss) and optimizer (Adam) for multi-class classification.


### STEP 5: 
Train the neural network using the training data through forward pass, loss computation, and backpropagation.


### STEP 6: 

est the trained model on unseen data and predict the customer segment (A, B, C, or D).



## PROGRAM

### Name:m.sahithi

### Register Number:212224040208
```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
      for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


```

### Dataset Information
<img width="1325" height="285" alt="image" src="https://github.com/user-attachments/assets/ee2e951a-ac5c-4be7-b6d7-5e3373419863" />

### OUTPUT
## Confusion Matrix
<img width="837" height="607" alt="image" src="https://github.com/user-attachments/assets/592dc1e0-a1c8-400a-95c1-83db213e2d29" />
## Classification Report
<img width="1501" height="417" alt="image" src="https://github.com/user-attachments/assets/5085dcaa-cc4e-4f66-86b4-5278de74fb9c" />

### New Sample Data Prediction
<img width="578" height="122" alt="image" src="https://github.com/user-attachments/assets/65ac2aba-f712-46e2-9b57-bf78c4d25bf0" />

## RESULT
<img width="1042" height="73" alt="image" src="https://github.com/user-attachments/assets/c7a6d59f-4590-46c0-b9e5-a261af754dc5" />
