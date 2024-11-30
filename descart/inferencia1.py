import cv2
import torch
import numpy as np
from modelo import CNN

# Mapeo de clases (asegúrate de que coincidan con las clases de entrenamiento)
CLASES = {0: "Cuadrado", 1: "Circulo", 2: "Tringulo"}

def preprocesar_imagen(img):
    img = cv2.resize(img, (28, 28))  # Redimensiona la imagen a 28x28 píxeles
    img = img.flatten() / 255.0  # Convierte la imagen en un vector unidimensional y normaliza los valores
    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28)

def contar_figuras(model, img, frame):
    """
    Detecta y cuenta figuras en una imagen binaria y dibuja rectángulos y etiquetas en el frame original.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encuentra contornos externos
    count = 0  # Contador de figuras detectadas correctamente

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Encuentra el rectángulo delimitador del contorno
        if w > 10 and h > 10:  # Filtra ruido (contornos muy pequeños)
            figura = img[y:y+h, x:x+w]  # Extrae la región de interés (ROI)
            entrada = preprocesar_imagen(figura)  # Preprocesa la imagen
            prediccion = model(entrada)  # Realiza la predicción con el modelo
            clase = torch.argmax(prediccion).item()  # Obtiene la clase predicha
            tipo_figura = CLASES.get(clase, "Desconocido")  # Traduce la clase al nombre de la figura

            # Dibuja el rectángulo y etiqueta en el frame original
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Rectángulo verde
            cv2.putText(frame, tipo_figura, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            count += 1  # Incrementa el contador si una figura válida fue detectada
    return count

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("modelo_entrenado.pth", weights_only=True))  # Carga los pesos del modelo entrenado
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el cuadro a escala de grises
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Umbraliza la imagen (figuras negras)

        # Detección y conteo de figuras
        count = contar_figuras(model, thresh, frame)

        # Muestra el número de figuras detectadas en el cuadro de video
        cv2.putText(frame, f"Figuras detectadas: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Detección de Figuras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Finaliza si se presiona 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
