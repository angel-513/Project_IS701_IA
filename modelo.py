import torch
import torch.nn as nn #Proporciona bloques de codigo para definir capas
import torch.nn.functional as F #Contiene las funciones de activacion

class CNN(nn.Module): #Hereda de nn.Module que contiene modulos de PyTorch para la construccion de modelos de RNAs
    def __init__(self):
        super(CNN, self).__init__() #Llama al constructor de la clase padre
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) #Parametros: numero de canales de entrada (una imagen en escala de grises tiene 1 canal), numero de mapas de caracteristicas que genera esta capa, kernel_size es el tamano de cada filtro, stride es la zancada del filtro, padding se agrega un borde de 1 pixel a la imagen para delimitarla
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) #Igual que la primera capa de convulusion, pero ahora toma 16 canales de entrada (los 16 generados por la capa anterior), y produce 32 canales de salida
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #Realiza subsampling (reduccion de tamano) para tomar unicamente las caracteristicas mas detacadas de la imagen. Lo hace por medio de matrices de 2x2
        self.fc1 = nn.Linear(64 * 3 * 3, 128) #Capa que recibe 64 mapas de caracteristicas de 3x3 generando 128 neuronas de salida
        self.fc2 = nn.Linear(128, 3) #Capa que recibe 128 valores de entrada y lo reduce a 3 clases de salida (3 tipos de figuras)
    
    def forward(self, tensor): #Funcion para definir el flujo de datos
        tensor = self.pool(F.selu(self.conv1(tensor))) #Pasa el tensor a traves de la primera capa convolucional generando 16 mapas de caracteristicas. Usa la funcion de activacion SELU (Scaled Exponential Linear Unit). Al pasar el resultado por la capa pool se reducen las dimensiones de cada mapa
        tensor = self.pool(F.selu(self.conv2(tensor))) #Pasa el resultado de la anterior por la segunda capa convolusional generando 32 mapas de caracteristicas. Usa SELU y reduce el tamano de los mapas con la capa pool
        tensor = self.pool(F.selu(self.conv3(tensor)))
        tensor = tensor.view(-1, 64 * 3 * 3) #Convierte los mapas en vectores lineales
        tensor = F.selu(self.fc1(tensor)) #Pasa los resultados por la primera capa totalmente conectada activada por SELU
        tensor = self.fc2(tensor) #Genera predicciones finales con 3 valores (uno por cada tipo de figura)

        return tensor