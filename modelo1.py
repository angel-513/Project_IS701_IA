import torch.nn as nn
import torch.nn.functional as F

class ModeloCNN(nn.Module):
    def __init__(self):
        super(ModeloCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Más filtros
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce dimensiones
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Más capas
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Una capa más profunda
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),  # Ajustado al tamaño del feature map final
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Regularización para evitar sobreajuste
            nn.Linear(128, 3)  # Tres clases: cuadrados, círculos, triángulos
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    model = ModeloCNN()
    print(model)
