import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()

        # Define the layers
        self.hidden = nn.Linear(1, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 100)
        self.hidden4 = nn.Linear(100, 100)
        self.hidden5 = nn.Linear(100, 100)

        # Batch normalization layers
        self.hidden_bn1 = nn.BatchNorm1d(100)
        self.hidden_bn2 = nn.BatchNorm1d(100)
        self.hidden_bn3 = nn.BatchNorm1d(100)
        self.hidden_bn4 = nn.BatchNorm1d(100)
        self.hidden_bn5 = nn.BatchNorm1d(100)

        # Output layer
        self.output = nn.Linear(100, 1)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.activation(self.hidden_bn1(self.hidden(x)))
        x = self.activation(self.hidden_bn2(self.hidden2(x)))
        x = self.activation(self.hidden_bn3(self.hidden3(x)))
        x = self.activation(self.hidden_bn4(self.hidden4(x)))
        x = self.activation(self.hidden_bn5(self.hidden5(x)))
        x = self.output(x)
        return x


# Define the training function
def train(x, y, model_name="models/SineApproximator.pt", epochs=10000):
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    model = SineApproximator().to(device)

    # Weight initialization function
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    model.apply(init_weights)  # Apply the weight initialization

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')

        scheduler.step(loss)  # Adjust the learning rate if necessary

    torch.save(model.state_dict(), model_name)

# Define the dataset generation function
def defineDataset(samples=100000):
    x = np.linspace(0, 2 * np.pi, samples)
    return x, np.sin(x)


def runInference(number: float, model_path="models/SineApproximator1.pt"):
    input_tensor = torch.tensor([[number]], dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    model = SineApproximator().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_value = output_tensor.cpu().numpy().item()

    return output_value


if __name__ == '__main__':
    x, y = defineDataset()
    #train(x, y)
    test_value = 0 #
    prediction = runInference(test_value)
    print(f'Prediction for sin({test_value}): {prediction:.3f}')
