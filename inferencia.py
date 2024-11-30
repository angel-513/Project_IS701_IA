import cv2
import torch
import numpy as np
from modelo import CNN

def preprocesar_imagen(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convierte la imagen captada por camara en escala de grises
    img = cv2.resize(img, (28, 28)) #Redimensiona la imagen a 28x28
    img = img.flatten() / 255.0 #Convierte la imagen en un vector lineal y normaliza los elementos

    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28) #Retorna la imagen convertida en tensor

def contar_figuras(model, img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Encuentra contornos externos en la imagen binaria
    count = 0 #COntador de figuras

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) #Encuentra el rectangulo que encierra cada contorno 
        figura = img[y:y+h, x:x+w] #Extrae la region de la imagen correspondiente al contorno
        entrada = preprocesar_imagen(figura) # Preprocesa la región de la imagen para que sea compatible con el modelo
        prediccion = model(entrada) #Realiza la prediccion con el modelo
        clase = torch.argmax(prediccion).item() #Obtiene la clase predicha (indice de la salida con mayor probabilidad)
        count += 1 #Incrementa el contador

        print(f"Figura detectada: {clase}") #Imprime la clase detectada
    
    return count #Devuelve el numero de figuras detectadas

if __name__ == "__main__":
    model = CNN()
    #model.load_state_dict(torch.load("modelo_entrenado.pth")) #Carga los pesos del modelo entrenado
    model.load_state_dict(torch.load("modelo_entrenado.pth", weights_only=True))
    model.eval() #Pone el modelo en modo de evaluacion (desactiva dropout y batch normalization). Metodo heredado

    cap = cv2.VideoCapture(0) #Abre la camara
    while True:
        ret, frame = cap.read() #Captura un cuadro de la camara
        if not ret:
            break #Termina el bucle si no se puede capturar un cuadro
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convierte el cuadro a escala de grises
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV) #Determina un umbral sobre la imagen para obtener una binaria (figuras negras sobre fondo blanco)
        count = contar_figuras(model, thresh)

        cv2.putText(frame, f"Figuras: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #Escribe el numero de figuras en el cuadro de video
        cv2.imshow("Deteccion de figuras", frame) #Muestra el cuadro procesado en una ventana

        if cv2.waitKey(1) & 0xFF == ord('q'): #Finaliza el programa si se presiona la tecla 'q
            break

    cap.release() #Libera la camara
    cv2.destroyAllWindows() #Cierra todas las ventanas de OpenCV
