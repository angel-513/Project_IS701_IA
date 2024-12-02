import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from modelo import CNN
import torchvision.transforms as transforms

def cargar_datos(csv_file):
    data = pd.read_csv(csv_file, header=None).values #Carga el archivo CSV
    X = data[:, :-1].astype('float32').reshape(-1, 1, 28, 28) #Toma las caracteristicas de las imagenes (valores de los pixeles) y los convierte a valores de punto flotante. Representa cada imagen en escala de grises con valores normalizados (entre 0 y 1)
    y = data[:, -1].astype('int64') #Contiene la columna de las etiquetas (las que dicen que tipo de figura son) y las transforma a enteros

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    X = torch.tensor(X)
    y = torch.tensor(y) #Convierte las caracteristicas y las etiquetas en tensores de PyTorch
    dataset = TensorDataset(X, y)

    return dataset

def guardar_pesos_txt(model, epoch):
    with open("pesos_por_epoca.txt", "a") as file:
        file.write(f"Pesos de la epoca: {epoch+1}:\n")
        for name, param in model.state_dict().items():
            file.write(f"{name}: \n{param.numpy()}\n")
        file.write("\n")

def entrenar_rna(model, train_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr) #Inicializa el optimizador Adam con los parametros del modelo. Ajusta los pesos del modelo para reducir la funcion de perdida
    criterion = nn.CrossEntropyLoss() #Funcion de perdida para clasificacion multiclase. Calcula la diferencia entre las predicciones y las etiquedas reales, o sea, la perdida.
    model.train() #Pone al modelo en modo de entrenamiento. Metodo heredado

    open("pesos_por_epoca.txt", "w").close()

    for epoch in range(epochs): #Itera por el numero de epocas
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for X_batch, y_batch in train_loader: #Recorre los lotes del DataLoader
            optimizer.zero_grad() #Limpia los gradientes acumulados
            outputs = model(X_batch) #Pasa el lote por la red para obtener predicciones
            loss = criterion(outputs, y_batch) #Calcula la perdida entre las predicciones y las etiquetas reales
            loss.backward() #Calcula los gradientes mediante backpropagation
            optimizer.step() #Actualiza los pesos del modelo
            
            epoch_loss += loss.item() #Acumula la perdida del lote

            #Calculo de precision
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

        accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoca {epoch+1}  Perdida: {epoch_loss:.4f}, Precision: {accuracy:.2f}%") #Imprime la perdida total de la epoca

        guardar_pesos_txt(model, epoch)

    torch.save(model.state_dict(), "modelo_entrenado_final.pth") #Guarda los pesos del modelo entrenado en un archivo
    print("Modelo guardado en modelo_entrenado_final.pth")

if __name__ == "__main__":
    dataset = cargar_datos("figuras.csv")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN()
    entrenar_rna(model, train_loader, epochs=50, lr=0.001)