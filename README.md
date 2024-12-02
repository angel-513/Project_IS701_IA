# PROYECTO INTELIGENCIA ARTIFICIAL IS701 - RED NEURONAL ARTIFICIAL CONVOLUCIONAL

El proyecto consiste en hacer una red neuronal convolucional que logre identificar figuras geometricas, tres tipos de figuras geometricas especificamente: circulos, cuadrados y triangulos. Para ello se usaron imagenes de las figuras dibujadas a mano y convertidas a imagenes de 28x28 pixeles. Las imagenes se extrajeron del siguiente repositorio: [cnn-with-pytorch](https://github.com/PeppeSaccardi/cnn-with-pytorch/tree/master).

## PREPROCESAMIENTO

### preprocesamiento.py

El preprocesamiento toma las imagenes, las convierte en matrices y luego en vectores lineales los cuales guarda en un archivo CSV. Los valores numericos de los vectores son los valores en escala de grises de cada pixel de cada imagen normalizados.

#### Librerias para preprocesamiento de imagenes

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

#### Funcion procesar_imagenes()

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

#### Funcion principal

```python
if __name__ == "__main__":
    procesar_imagenes("./images", "figuras.csv")
```

Simplemente llama a la función procesar_imagenes() pansadole como parametros el directorio de la carpeta de imagenes y el nombre del CSV de salida.

## MODELADO DE LA RNA CONVOLUCIONAL

### modelo.py

En este archivo se contruye la arquitectura o el modelo de la red neuronal artificial con capas totalmente conectadas y con capas convolucionales.

#### Librerias de modelado de red neuronal

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

* La libreria ***torch*** es la principal del proyecto, es la que proporciona las funciones para crear las capas de la red neuronal y las funciones de activacion.

* **torch.nn** es un modulo de PyTorch, en este caso se usa para la creacion de capas del modelo.

* **torch.nn.functional** es el modulo de PyTorch que contiene las funciones de activacion disponibles para el entrenamiento de la RNA.

#### Clase CNN

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

##### Constructor

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

##### Flujo de datos

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

## ENTRENAMIENTO DE LA RNA

### entrenamiento.py

Este archivo carga los datos, entrena a la red neuronal e imprime el error y los pesos de cada epoca de entrenamiento.

#### Librerias para el entrenamiento de la RNA

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

* ***torch.optim*** es un modulo de PyTorch que contiene optimizadores, como Adam, el utilizado en este proyecto, que ajustan los pesos del modelo durante el entrenamiento.

* ***torch.nn**

* ***pandas***

* ***torch.utils.data*** de este modulo se extraen las funciones *DataLoader* y *TensorDataset* que ayudan a manejar los datos y a dividirlos en lotes durante el entrenamiento. De este modulo tambien se extrae la clase *Dataset* para utilizarla como clase padre de la clase auxiliar *FigurasDataset*.

* ***modelo***

* ***torch.transforms*** incluye transformaciones para datos como rotaciones, normalización, etc.

#### Funcion para cargar los datos

```python
def cargar_datos(csv_file):
    data = pd.read_csv(csv_file, header=None).values
    X = data[:, :-1].astype('float32').reshape(-1, 1, 28, 28)
    y = data[:, -1].astype('int64')

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    X = torch.tensor(X)
    y = torch.tensor(y)
    dataset = TensorDataset(X, y)

    return dataset
```

Es la funcion que se carga de leer los datos del archivo CSV generado en el preprocesamiento. El metodo ***pd.read_csv*** lee los datos desde el archivo. Se definen las variables *X* y *y*, las cuales contienen los valores de los pixeles de las imagenes y la etiqueta (clases o tipos de figura) de cada una, respectivamente. Por ultimo, se transforman esas variables en tensores y se usan esos tensores para construir un set de datos el cual se retorna.

#### Funcion para guardar los pesos

```python
def guardar_pesos_txt(model, epoch):
    with open("pesos_por_epoca.txt", "a") as file:
        file.write(f"Pesos de la epoca: {epoch+1}:\n")
        for name, param in model.state_dict().items():
            file.write(f"{name}: \n{param.numpy()}\n")
        file.write("\n")
```

Es una funcion para guardar los pesos y sesgos (bias) de cada epoca de entrenamiento en un archivo de texto plano. El metodo *state_dict()* extrae los pesos del modelo y el metodo *items* itera sobre ellos. Luego se convierten los pesos a un formato numpy para que sean legibles.

#### Funcion de entrenamiento

```python
def entrenar_rna(model, train_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    open("pesos_por_epoca.txt", "w").close()

    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            #Calculo de precision
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

        accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoca {epoch+1}  Perdida: {epoch_loss:.4f}, Precision: {accuracy:.2f}%")

        guardar_pesos_txt(model, epoch)

    torch.save(model.state_dict(), "modelo_entrenado_final.pth")
    print("Modelo guardado en modelo_entrenado_final.pth")
```

Esta es la funcion principal, al inicar se define un optimizador basado en el algoritmo Adam (*Adaptive Moment Estimation*), el cual ajusta automaticamente la tasa de aprendizaje durante el entrenamiento segun el promedio de los gradientes. Tambien se define la funcion de perdida, utilizamos la funcion *CrossEntropyLoss* proporcionada por el modulo **nn** de PyTorch, esta es una funcion comunmente usada en problemas de clasificacion de multiples clases, en pocas palabras, mide cuan lejos estan las predicciones del modelo de las etiquetas verdaderas.
Luego se pone al modelo en modo de entrenamiento.

Se comienza con la iteración de las epocas ingresadas, se definen tres variables: *epoch_loss* que guadara la perdida de cada epoca, *correct_predictions* que es un contador de cuantas predicciones se hicieron correctamente, y *total_predictions* que es un contador de predicciones totales.

Por cada epoca se recorre cada lote de datos del dataset ingresado para el entrenamiento. Se limpian los gradientes acumulados del lote anterior (*optimizer.zero_grad*), se le pasa el lote de datos (sin etiquetas) al modelo y se guardan las salidas en la variable *outputs*, se comparan las salidas con el lote de etiquetas para calcular la perdida entre las predicciones y los resultados esperados, se calculan los gradientes mediante **Backpropagation** y se actualizan los pesos (*optimize.step*). Para terminar la iteracion del lote se guarda la perdida de la epoca y se hace el calculo del precision.

Para terminar con la iteracion de la epoca se calcula el porcentaje de precision y se imprime junto con la perdida de la epoca. Por ultimo se guardan los pesos de esa epoca en el archivo de texto plano.

Al terminar con todas las epocas, se guarda el modelo entrenado.

#### Funcion principal del entrenamiento

Se cargan los datos de entrenamiento, se crea el set de datos compatible con PyTorch, se instancia la clase del modelo, y se ejecuta la funcion de entrenamiento pasandole como parametros el modelo, el set de datos, el numero de epocas y la tasa de aprendizaje inicial.

## DETECCION DE FIGURAS POR CAMARA

### inferencia4.py

Este archivo realiza detección de figuras geométricas (círculos, cuadrados y triángulos) en tiempo real utilizando una cámara web y el modelo entrenado previamente con PyTorch.

#### Librerias de inferencia

```python
import cv2
import torch
import numpy as np
from modelo import CNN
```

#### Definicion de clases (tipos de figuras)

```python
CLASES = {0: "Circulo", 1: "Cuadrado", 2: "Triangulo"}
```

#### Funcion de preprocesamiento de imagenes

```python
def preprocesar_imagen(img):
    img = cv2.resize(img, (28, 28))
    img = img.flatten() / 255.0
    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28)
```

Lo que hace la funcion es tomar una image y asegurar se su tamano redimensionandola a 28x28 pixeles, y luego aplanando la matriz obtenida en un vector lineal el cual se pasa por la funcion *tensor* para convertirlo en un tensor de PyTorch con el formato adecuado (1 canal de color, dimensiones de 28x28).

#### Funcion de conteo de figuras

```python
def contar_figuras(model, img, frame):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    contornos_procesados = []

    for i, contour in enumerate(contours):
        # Ignorar contornos internos
        if hierarchy[0][i][3] != -1:  # Tiene un "padre" (es interno)
            continue

        area = cv2.contourArea(contour)
        if area < 500:  # Filtra áreas pequeñas (ruido)
            continue

        # Verificar solapamiento con contornos ya procesados
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box_actual = (x, y, x + w, y + h)
        solapado = False

        for bbox in contornos_procesados:
            x1, y1, x2, y2 = bbox
            if (
                x >= x1 and y >= y1 and x + w <= x2 and y + h <= y2
            ) or (x1 >= x and y1 >= y and x2 <= x + w and y2 <= y + h):
                solapado = True
                break

        if solapado:
            continue

        # Agregar bounding box al conjunto procesado
        contornos_procesados.append(bounding_box_actual)

        figura = img[y:y+h, x:x+w]

        entrada = preprocesar_imagen(figura)
        prediccion = model(entrada)
        clase = torch.argmax(prediccion).item()
        tipo_modelo = CLASES.get(clase, "Desconocido")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, tipo_modelo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        count += 1
    return count
```

Se le pasa el modelo, una imagen y un frame (ventana de captura de imagen) como parametros. La funcion *findContours* encuentra los contornos en la imagen binarizada (la imagen pasada como parametro) y devuelve una lista de contornos y su jerarquia (relacion entre los contornos internos y externos). La variable *count* es un contador que cuenta las figuras detectadas. El arreglo *contornos_procesados* guarda los contornos que ya han sido procesados.

Luego se inicia un bucle que itera sobre cada contorno detectado. Se verifica si el contorno tiene un padre, es decir, si es un contorno interno o externo, si lo tiene se ignora el contorno.

Se calcula un area del contorno y filtra las areas pequenas consideradas como ruido.

Luego verifica solapamiento entre contornos. La funcion *boundingRect* calcula el rectangulo delimitador del contorno con coordenadas **(x,y)** y de tamano **w x h**. Luego se verifica si el contorno en procesamiento se solapa con los contornos procesados anteriormente utilizando una relacion entre rectangulos delimitadores. Por ultimo se agrega el contorno procesado al arreglo *contornos_procesados*.

Posterior al procesamiento del contorno, se extrae la region de la figura detectada de la imagen original y se le manda a la funcion *preprocesar_imagen*, el resultado de la funcion se le pasa al modelo como entrada para obtener una prediccion. La funcion *argmax* de PyTorch obtiene la clase con la mayor probabilidad, la cual se presentara como prediccion final. Se mapea el diccionario de clases definido al principio y se clasifica la prediccion en una de las tres clases posibles.

Por ultimo, se dibuja un rectangulo en pantalla que encierre a la figura detectada y se imprime la prediccion del modelo encima del rectangulo. El contador de figuras *count* suma uno por cada figura detectada y se retorna como resultado de la funcion.

#### Funcion principal de inferencia

```python
if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("modelo_entrenado_final.pth", weights_only=True))
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        count = contar_figuras(model, thresh, frame)

        cv2.putText(frame, f"Figuras detectadas: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Deteccion de Figuras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

Carga el modelo preentrenado (*modelo_entrenado_final.pth*). Selecciona el dispositivo de captura de video y lo inicia. Se toma el frame y se convierte a escala de grises, se aplica un desenfoque y un umbral adaptativo para binarizar la imagen. Se aplican operaciones morfologicas para mejorar la imagen. Se llama la funcion *contar_figuras* para detectar las figuras y obtener su cantidad, se le pasan el modelo, la imagen binarizada y el frame de captura como parametros. Por ultimo se imprime en pantalla una leyenda que muestra el conteo de figuras y se abre la ventana de video.

La ventana se cierra al presionar la tecla "q". Finalmente, libera el dispositivo (la camara) y cierra la ventada de video.
