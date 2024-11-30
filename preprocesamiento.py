import os
import cv2
import numpy as np
import pandas as pd

def procesar_imagenes(input_dir, output_csv):
    data = []
    labels = {
        'circulo': 0,
        'cuadrado': 1,
        'triangulo': 2
    }

    for label, value in labels.items():
        path = os.path.join(input_dir, label) #Path de cada carpeta de imagenes

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name) #Path de cada imagen
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Carga de las imagenes en escala de grises
            img = cv2.resize(img, (28,28)) #Redimension de las imagenes a 28x28 pixeles
            img_flattened = img.flatten() / 255.0 #img.flatten la matriz de 28x28 en un vector lineal de 784 elementos. La division entre 255.0 pasa los elementos a 0 o 1
            data.append(np.append(img_flattened, value)) #Se agrega el vector de la imagen al set de datos, los 784 elementos de la imagen + 1 que corresponde al valor de la etiqueta (el tipo de figura)

    df = pd.DataFrame(data) #Se crea un DataFrame (estructura de datos) con la data de las imagenes convertidas en vectores
    df.to_csv(output_csv, index=False, header=False) #Se convierte el DataFrame en un archivo CSV
    print(f"Datos guardados en {output_csv}")

if __name__ == "__main__":
    procesar_imagenes("./images", "figuras.csv")