import cv2
import torch
import numpy as np
from modelo import CNN

CLASES = {0: "Cuadrado", 1: "Circulo", 2: "Triangulo"}

def preprocesar_imagen(img):
    img = cv2.resize(img, (28, 28))
    img = img.flatten() / 255.0
    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28)

def detectar_geometria(contorno):
    epsilon = 0.04 * cv2.arcLength(contorno, True)
    aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
    if len(aproximacion) == 4:
        return "Cuadrado"
    elif len(aproximacion) > 8:
        return "Circulo"
    else:
        return "Otro"

def contar_figuras(model, img, frame):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filtra ruido
            x, y, w, h = cv2.boundingRect(contour)
            figura = img[y:y+h, x:x+w]

            tipo_geometrico = detectar_geometria(contour)

            entrada = preprocesar_imagen(figura)
            prediccion = model(entrada)
            clase = torch.argmax(prediccion).item()
            tipo_modelo = CLASES.get(clase, "Desconocido")

            # Resolución final
            if tipo_geometrico == tipo_modelo:
                tipo_figura = tipo_geometrico
            else:
                tipo_figura = f"{tipo_modelo}/{tipo_geometrico}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, tipo_figura, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            count += 1
    return count

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("modelo_entrenado.pth", weights_only=True))
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV)

        count = contar_figuras(model, thresh, frame)

        cv2.putText(frame, f"Figuras detectadas: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Detección de Figuras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
