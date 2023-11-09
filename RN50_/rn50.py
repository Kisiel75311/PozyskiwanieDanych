import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# exit(0)

# Ścieżki do folderów z danymi
path_train = "archive(12)/chest_xray/train"
path_test = "archive(12)/chest_xray/test"
path_val = "archive(12)/chest_xray/val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Wczytaj zbiory danych
train_dataset = ImageFolder(path_train, transform=transform)
test_dataset = ImageFolder(path_test, transform=transform)
val_dataset = ImageFolder(path_val, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optymalizator i funkcja straty
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

import torch

# ...

epochs = 10  # Możesz dostosować liczbę epok

for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {total_loss / len(train_loader)}")

    # Walidacja
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss / len(val_loader)} - Val Accuracy: {correct / total:.2f}")

# Testowanie
# Ocenianie modelu na zbiorze testowym przy użyciu metryki MCC
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

mcc = matthews_corrcoef(true_labels, predicted_labels)
print(f"MCC on Test Set: {mcc:.4f}")
