import torch
import torch.nn as nn
import torch.optim as optim

# definiowanie modelu
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(299 * 512, 50)  # fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flattening the input matrix
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)

model = NeuralNetwork()# tworzenie modelu
# funkcja kosztu (CrossEntropyLoss dla klasyfikacji wieloklasowej)
criterion = nn.CrossEntropyLoss()
# optymalizator 
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #gradientowy, lr- wsp. uczenia, momentu - dynamika w optymalizacji
optimizer = optim.Adam(model.parameters(), lr=0.001)
# przykładowe dane treningowe
input_data = torch.randn(100, 299, 512)  # 100 przykładów
target = torch.randint(0, 50, (100,))

# uczenie modelu
num_epochs = 100
for epoch in range(num_epochs):
  #forward pass
    output = model(input_data)
    loss = criterion(output, target) #obliczanie f. kosztu na podstawie przewidywań podelu i rzeczywistych etykiet
  #backward pass
    optimizer.zero_grad() #czyszczenie gradientów
    loss.backward() #gradienty f. kosztu wzlędem wag
    optimizer.step() #aktualizowanie wag modelu na podstawie obliczonych gradientów

    #aktualnego stanu uczenia co 10 iteracji
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Przykładowe dane testowe
test_data = torch.randn(10, 299, 512)  # 10 przykładów do testowania

# Przewidywanie na danych testowych
with torch.no_grad():
    model.eval()
    predictions = model(test_data)
    _, predicted_labels = torch.max(predictions, 1)

    print("\nPredicted Labels:")
    print(predicted_labels.numpy())
