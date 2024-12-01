# PREPROCESAMIENTO DE DATOS

El proyecto consiste en hacer una red neuronal convolucional que logre identificar figuras geometricas, tres tipos de figuras geometricas especificamente: circulos, cuadrados y triangulos. Para ello se usaron imagenes de las figuras dibujadas a mano y convertidas a imagenes de 28x28 pixeles. Las imagenes se extrajeron del siguiente repositorio: [cnn-with-pytorch](https://github.com/PeppeSaccardi/cnn-with-pytorch/tree/master).

## preprocesamiento.py

El preprocesamiento toma las imagenes, las convierte en matrices y luego en vectores lineales los cuales guarda en un archivo CSV. Los valores numericos de los vectores son los valores en escala de grises de cada pixel de cada imagen normalizados.

### Librerias para preprocesamiento de imagenes

```python
import os
import cv2
import numpy as np
import pandas as pd
```

* La libreria ***os*** se utiliza para acceder a directorios (carpetas) e iterar sobre los archivos de esos directorios, imagenes en este caso.

* La libreria ***cv2*** se usa para manipular imagenes y videos. En este archivo se usa para la manipulacion de las imagenes de las figuras geometricas (convertirlas a escala de grises y redimensionarlas).

* La libreria ***numpy*** en este caso se usa para manipular los vectores lineales obtenidos despues del analisis de las imagenes.

* La libreria ***pandas*** es para manipulación de datos, en este archivo se usa la estructura de datos DataFrame que ofrece para almacenar los vectores que posteriormente se convierte en el archivo CSV.

### Funcion procesar_imagenes()

```python
def procesar_imagenes(input_dir, output_csv):
    data = []
    labels = {
        'circulo': 0,
        'cuadrado': 1,
        'triangulo': 2
    }

    for label, value in labels.items():
        path = os.path.join(input_dir, label) 

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28,28))
            img_flattened = img.flatten() / 255.0
            data.append(np.append(img_flattened, value))

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Datos guardados en {output_csv}")
```

Es la unica funcion del archivo y es la que procesa las imagenes de entrada con la que se entrenaran al modelo de la red neuronal. Define la variable ***data*** como un arreglo, en esta se guardaran los vectores lineales. Define el diccionario ***labels*** para etiquetar cada figura con su correspondiente clase.

El primer ciclo *for* es para recorrer cada carpeta de imagenes, el segundo ciclo es para recorrer cada imagen dentro de las carpetas pasarlas por un filtro que las convierte a escala de grises (un solo canal cromatico), luego se asegura de que sean imagenes de 28x28 redimensionandolas obteniendo matrices de valores numericos que se aplana para obtener los vectores lineales.

Al salir de los ciclos, se crea la estructura **DataFrame** que guarda todos los vectores y se convierte en un archivo CSV sin encabezados.

### Funcion principal

```python
if __name__ == "__main__":
    procesar_imagenes("./images", "figuras.csv")
```

Simplemente llama a la función procesar_imagenes() pansadole como parametros el directorio de la carpeta de imagenes y el nombre del CSV de salida.

# MODELADO DE LA RNA CONVOLUCIONAL

## modelo.py

En este archivo se contruye la arquitectura o el modelo de la red neuronal artificial con capas totalmente conectadas y con capas convolucionales.

### Librerias de modelado de red neuronal

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

* La libreria ***torch*** es la principal del proyecto, es la que proporciona las funciones para crear las capas de la red neuronal y las funciones de activacion.

* **torch.nn** es un modulo de PyTorch, en este caso se usa para la creacion de capas del modelo.

* **torch.nn.functional** es el modulo de PyTorch que contiene las funciones de activacion disponibles para el entrenamiento de la RNA.

### Clase CNN

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 3)
    
    def forward(self, tensor):
        tensor = self.pool(F.selu(self.conv1(tensor)))
        tensor = self.pool(F.selu(self.conv2(tensor)))
        tensor = self.pool(F.selu(self.conv3(tensor)))
        tensor = tensor.view(-1, 64 * 3 * 3)
        tensor = F.selu(self.fc1(tensor))
        tensor = self.fc2(tensor)

        return tensor
```

La clase CNN define la arquitectura o modelo de la RNA, hereda del modulo nn de PyTorch (*nn.Module*).

#### Constructor

El constructor de la clase (**\_\_init\_\_**) llama al constructor de la clase padre para inicializar la clase. Luego define las capas de la red neuronal:

* ***conv1*** es una capa convolucional 2D que ayuda a extraer caracteristicas espaciales de las imagenes y patrones locales como bordes, texturas o formas.

    Los parametros que tiene son:
  * *in_channels*: el numero de canales de entrada (1 para escala de grises).
  * *out_channels*: numero de filtros o caracteristicas que aprendera la capa y que se convertiran en entradas de la siguiente capa.
  * *kernel_size*: tamano del filtro (matriz 3x3).
  * *stride*: es el desplazamiento o zancada del filtro.
  * *padding*: le da un borde a la imagen para mantener el tamano y evitar perder informacion.

* ***conv2*** igual, es una capa convolucional que genera un mapa de características que destaca patrones detectados por los filtros.

* ***conv3*** una tercera capa convolucional que fue agregada para mejorar la prediccion de la RNA.

* ***pool*** es una capa para aplicar submuestreo (reduccion de tamano) a los mapas de características. Divide el mapa en ventanas y capta las características mas importantes.

    Al igual  que las capas convolucionales, se le indica un tamano de filtro (*kernel_size*) y el desplazamiento del filtro (*stride*).

* ***fc1*** es una capa totalmente conectada que realiza un transformacion lineal en los datos de entrada:

    ![Fórmula](https://latex.codecogs.com/png.latex?y%20%3D%20xW%5ET%20%2B%20b)

* ***fc2*** es otra capa totalmente conectada, se utiliza principalmente para conectar características extraídas en capas anteriores (las convolucionales) con las salidas de la red, las predicciones finales.

#### Flujo de datos

La función ***forward*** es para definir el flujo de los datos, recibe un tensor como parametro. Se usa la función de activacion SELU (*Scaled Exponential Linear Unit*), es una función que permite que las salidas de las neuronas tengan valores adecuados al propagarse por la red:

\[
\text{SELU}(x) =
\begin{cases}
\lambda x & \text{si } x > 0, \\
\lambda \alpha (e^x - 1) & \text{si } x \leq 0,
\end{cases}
\]

Donde:

* \(\alpha \approx 1.673\)
* \(\lambda \approx 1.050\)

Si el valor de entrada es positivo la salida crece de forma lineal. Pero si es negativo se usa una curva exponencial que suaviza los valores. Esto ayuda a que la red ajuste mejor los pesos.

El flujo es el siguiente:

1. Pasa el tensor por la primera capa convolucional, le aplica la función SELU y luego pasa el resultado por la capa pool.

2. Pasa el tensor por la segunda capa convolucional, le aplica la función SELU y luego pasa el resultado por la capa pool.

3. Pasa el tensor por la tercera capa convolucional, le aplica la función SELU y luego pasa el resultado por la capa pool.

4. Despues de pasar por las capas convolucionales y de submuestreo, el tensor es un conjunto de matrices, asi que se aplanan en vectores lineales para ingresar a la capa totalmente conectada.

5. Pasa por la primera capa totalmente conectada y da como resultado un vector de 128 elemetos/caracteristicas.

6. Pasan por la segunda capa totalmente conectada que toma los 128 elemetos y los reduce a 3, las 3 clases de figuras geometricas posibles.

7. Finalmente se retorna el tensor.

# ENTRENAMIENTO DE LA RNA

## entrenamiento.py

Este archivo carga los datos, entrena a la red neuronal e imprime el error y los pesos de cada epoca de entrenamiento.

### Librerias para el entrenamiento de la RNA

```python
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from modelo import CNN
import torchvision.transforms as transforms
```

* ***torch***

* ***torch.optim***

* ***torch.nn**

* ***pandas***

* ***torch.utils.data***

* ***modelo***

* ***torch.transforms***
