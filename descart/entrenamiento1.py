import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from modelo1 import ModeloCNN
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

# Configuración inicial
BATCH_SIZE = 300
EPOCHS = 30
LEARNING_RATE = 0.0001

# Aumentación de datos
transformaciones = transforms.Compose([
    transforms.RandomRotation(30),  # Rotación aleatoria
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Recortes aleatorios
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brillo y contraste
    transforms.ToTensor()
])

# Dataset desde una carpeta
dataset = ImageFolder("./images", transform=transformaciones)

# Dividir en entrenamiento y validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inicializar el modelo
model = ModeloCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def entrenar(model, train_loader, val_loader, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # Ajustar el learning rate
        print(f"Época {epoch+1}/{epochs}, Pérdida: {running_loss/len(train_loader):.4f}")

        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validación - Pérdida: {val_loss/len(val_loader):.4f}, Precisión: {100 * correct/total:.2f}%")

# Guardar el modelo al final
entrenar(model, train_loader, val_loader, EPOCHS, device)
torch.save(model.state_dict(), "modelo_mejorado.pth")
print("Modelo guardado como 'modelo_mejorado.pth'.")
