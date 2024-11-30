import cv2
import torch
import numpy as np
from modelo import CNN

CLASES = {0: "Circulo", 1: "Cuadrado", 2: "Triangulo"}

def preprocesar_imagen(img):
    img = cv2.resize(img, (28, 28))
    img = img.flatten() / 255.0
    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28)

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
