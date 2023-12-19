import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(299 * 512, 50)  # fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flattening the input matrix
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)
    
language_labels = ['Polish', 'English', 'French', 'German', 'Spanish', 'Italian', 'Russian', 'Chinese', 'Japanese', 'Arabic', 
                   'Portuguese', 'Dutch', 'Swedish', 'Korean', 'Turkish', 'Hindi', 'Greek', 'Czech', 'Hungarian', 'Thai', 'Indonesian', 
                    'Norwegian', 'Romanian', 'Bulgarian', 'Croatian', 'Serbian', 'Slovak', 'Slovenian', 'Ukrainian', 'Estonian', 'Latvian', 'Lithuanian', 'Vietnamese', 'Malay', 'Tagalog' ]
input_size = 299*512
num_languages = len(language_labels)
model = Net()
learning_rate = 0.001
batch_size = 64
epochs = 100

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
input_data = torch.randn(64, input_size) 
targets = torch.randint(0, num_languages, (batch_size,)) 

# Dane walidacyjne
validation_data = torch.randn(32, input_size)
validation_targets = torch.randint(0, num_languages, (32,))

# Trening modelu
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    #if (epoch + 1) % 10 == 0:
     #   print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    # Walidacja modelu na danych walidacyjnych
    model.eval()  # Ustawiamy model w tryb ewaluacji
    with torch.no_grad():
        validation_output = model(validation_data)
        validation_loss = criterion(validation_output, validation_targets)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {validation_loss.item():.4f}')