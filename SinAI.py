import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()

        self.hidden = nn.Linear(1, 30)
        self.output = nn.Linear(30, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train(x, y, model_name="models/SineApproximator.pt", epochs=5000):
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    model = SineApproximator().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        y_pred = model(x_tensor)

        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')

            model.eval()
            with torch.no_grad():
                y_test = model(x_tensor).cpu().numpy()
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label='True Sine', linewidth=2)
                plt.plot(x, y_test, label='Predicted Sine', linestyle='--')
                plt.legend()
                plt.title(f"Sine Function Approximation with ML")
                plt.xlabel("x")
                plt.ylabel("sin(x)")
                plt.close()

    torch.save(model.state_dict(), model_name)


def defineDataset(samples=10000):
    x = np.linspace(0, 2 * np.pi, samples)
    return x, np.sin(x)


def runInference(number: float, model_path="models/SineApproximator.pt"):
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
    test_value = np.pi / 2  #
    prediction = runInference(test_value)
    print(f'Prediction for sin({test_value}): {prediction}')
