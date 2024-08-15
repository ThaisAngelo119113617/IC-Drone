# import cv2
# import numpy as np

# # Função para detectar a base de takeoff ou pouso
# def detect_base(input_frame):
#     # Convertendo a imagem para escala de cinza
#     gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
#     # Aplicando um filtro de borda
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Encontrando contornos na imagem
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         # Aproximando o contorno para um polígono
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         # Se o polígono tiver 4 lados, pode ser uma base
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
            
#             # Verificando a razão de aspecto para diferenciar entre takeoff e pouso
#             if 0.9 < aspect_ratio < 1.1:
#                 cv2.putText(input_frame, "Takeoff Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 cv2.drawContours(input_frame, [approx], 0, (0, 255, 0), 2)
#             else:
#                 cv2.putText(input_frame, "Landing Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 cv2.drawContours(input_frame, [approx], 0, (0, 0, 255), 2)
   
#     return input_frame

# # Capturando vídeo da webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detectando a base na imagem capturada
#     frame = detect_base(frame)
    
#     # Mostrando a imagem com a detecção
#     cv2.imshow('Base Detection', frame)
    
#     # Saindo do loop ao pressionar 'q'
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Função para detectar a base de takeoff ou pouso
# def detect_base(input_frame, base_list):
#     # Convertendo a imagem para escala de cinza
#     gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
#     # Aplicando um filtro de borda
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Encontrando contornos na imagem
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         # Aproximando o contorno para um polígono
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         # Se o polígono tiver 4 lados, pode ser uma base
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
            
#             # Verificando a razão de aspecto para diferenciar entre takeoff e pouso
#             if 0.9 < aspect_ratio < 1.1:
#                 # Se a última base foi de takeoff, a próxima deve ser landing
#                 if len(base_list) == 0 or base_list[-1] == "Landing":
#                     cv2.putText(input_frame, "Takeoff Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     cv2.drawContours(input_frame, [approx], 0, (0, 255, 0), 2)
#                     base_list.append("Takeoff")
#             else:
#                 # Se a última base foi de landing, a próxima deve ser takeoff
#                 if len(base_list) == 0 or base_list[-1] == "Takeoff":
#                     cv2.putText(input_frame, "Landing Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                     cv2.drawContours(input_frame, [approx], 0, (0, 0, 255), 2)
#                     base_list.append("Landing")
   
#     return input_frame

# # Capturando vídeo da webcam
# cap = cv2.VideoCapture(0)

# # Lista para armazenar o histórico de bases detectadas
# base_list = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detectando a base na imagem capturada
#     frame = detect_base(frame, base_list)
    
#     # Mostrando a imagem com a detecção
#     cv2.imshow('Base Detection', frame)
    
#     # Saindo do loop ao pressionar 'q'
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import time

# # Função para detectar a base de takeoff ou pouso
# def detect_base(input_frame, base_list):
#     # Convertendo a imagem para escala de cinza
#     gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
#     # Aplicando um filtro de borda
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Encontrando contornos na imagem
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         # Aproximando o contorno para um polígono
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         # Se o polígono tiver 4 lados, pode ser uma base
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
            
#             # Verificando a razão de aspecto para diferenciar entre takeoff e pouso
#             if 0.9 < aspect_ratio < 1.1:
#                 # Se a última base foi de takeoff, a próxima deve ser landing
#                 if len(base_list) == 0 or base_list[-1] == "Landing":
#                     cv2.putText(input_frame, "Takeoff Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     cv2.drawContours(input_frame, [approx], 0, (0, 255, 0), 2)
#                     base_list.append("Takeoff")
#                     return "Takeoff"
#             else:
#                 # Se a última base foi de landing, a próxima deve ser takeoff
#                 if len(base_list) == 0 or base_list[-1] == "Takeoff":
#                     cv2.putText(input_frame, "Landing Base", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                     cv2.drawContours(input_frame, [approx], 0, (0, 0, 255), 2)
#                     base_list.append("Landing")
#                     return "Landing"
   
#     return None

# # Capturando vídeo da webcam
# cap = cv2.VideoCapture(0)

# # Lista para armazenar o histórico de bases detectadas
# base_list = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detectando a base na imagem capturada
#     detected_base = detect_base(frame, base_list)
    
#     # Mostrando a imagem com a detecção
#     cv2.imshow('Base Detection', frame)
    
#     # Se uma base for detectada, esperar 2 segundos antes de continuar
#     if detected_base is not None:
#         time.sleep(1)
    
#     # Saindo do loop ao pressionar 'q'
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def detect_base(input_frame):
#     # Convertendo a imagem para escala de cinza
#     gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
#     # Aplicando um filtro de suavização para reduzir ruídos
#     blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
#     # Detectando círculos na imagem
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
#                                param1=50, param2=30, minRadius=10, maxRadius=100)
    
#     # Detectando contornos
#     edges = cv2.Canny(blurred, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
        
#         for (x, y, r) in circles:
#             # Verificando uniformidade da cor dentro do círculo
#             mask = np.zeros_like(gray)
#             cv2.circle(mask, (x, y), r, 255, -1)
#             mean_val = cv2.mean(gray, mask=mask)[0]

#             # Aplicando threshold para determinar se o círculo é preenchido
#             if mean_val > 100:  # Ajustar esse valor conforme necessário
#                 label = "Takeoff Base"
#                 color = (0, 255, 0)
#             else:
#                 label = "Landing Base"
#                 color = (0, 0, 255)
            
#             # Desenhando o círculo e o rótulo na imagem
#             cv2.circle(input_frame, (x, y), r, color, 4)
#             cv2.putText(input_frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     # Processando os contornos detectados
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
#         if len(approx) == 4:  # Verificando se o contorno é um quadrado/retângulo
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             if 0.9 < aspect_ratio < 1.1:
#                 cv2.drawContours(input_frame, [approx], 0, (255, 255, 0), 2)

#     return input_frame

# # Capturando vídeo da webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detectando a base na imagem capturada
#     frame = detect_base(frame)
    
#     # Mostrando a imagem com a detecção
#     cv2.imshow('Base Detection', frame)
    
#     # Saindo do loop ao pressionar 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return edges

# def find_contours(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def detect_circle(image, contour):
#     mask = np.zeros_like(image)
#     cv2.drawContours(mask, [contour], -1, 255, -1)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
#     circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         return circles
#     return None

# def classify_base(image, contours):
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         if len(approx) == 4:  # Verificando se é um retângulo/quadrado
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             area = cv2.contourArea(contour)
            
#             if (1.3 < aspect_ratio < 1.7 and area > 30000):  # Base de Takeoff
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Takeoff Base"
#             elif (0.9 < aspect_ratio < 1.1 and area > 9000):  # Base de Pouso
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Landing Base"
    
#     return "Unknown Base"

# def main():
#     # Iniciar captura da webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Erro ao acessar a webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Erro ao capturar o frame.")
#             break
        
#         edges = preprocess_image(frame)
#         contours = find_contours(edges)
#         base_type = classify_base(frame, contours)
        
#         # Mostrar resultado na tela
#         cv2.putText(frame, f"Detected base: {base_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow("Frame", frame)
        
#         # Pressione 'q' para sair
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Executar a função principal
# main()


# import cv2
# import numpy as np

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return edges

# def find_contours(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def detect_circle(image, contour):
#     # Criar uma máscara do mesmo tamanho da imagem
#     mask = np.zeros(image.shape[:2], dtype="uint8")  # Garantir que a máscara seja em escala de cinza
#     cv2.drawContours(mask, [contour], -1, 255, -1)
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
#     circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         return circles
#     return None

# def classify_base(image, contours):
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         if len(approx) == 4:  # Verificando se é um retângulo/quadrado
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             area = cv2.contourArea(contour)
            
#             if (1.3 < aspect_ratio < 1.7 and area > 30000):  # Base de Takeoff
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Takeoff Base"
#             elif (0.9 < aspect_ratio < 1.1 and area > 9000):  # Base de Pouso
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Landing Base"
    
#     return "Unknown Base"

# def main():
#     # Iniciar captura da webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Erro ao acessar a webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Erro ao capturar o frame.")
#             break
        
#         edges = preprocess_image(frame)
#         contours = find_contours(edges)
#         base_type = classify_base(frame, contours)
        
#         # Mostrar resultado na tela
#         cv2.putText(frame, f"Detected base: {base_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow("Frame", frame)
        
#         # Pressione 'q' para sair
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Executar a função principal
# main()

# ################3
import cv2
import numpy as np

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return edges

# def find_contours(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def detect_circle(image, contour):
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     cv2.drawContours(mask, [contour], -1, 255, -1)
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
#     circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         return circles
#     return None

# def classify_base(image, contours):
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
#         if len(approx) == 4:  # Verificando se é um retângulo/quadrado
#             _, __, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             area = cv2.contourArea(contour)
            
#             if (1.3 < aspect_ratio < 1.7 and area > 30000):  # Base de Takeoff
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Takeoff Base"
#             elif (0.9 < aspect_ratio < 1.1 and area > 9000):  # Base de Pouso
#                 circle = detect_circle(image, approx)
#                 if circle is not None:
#                     return "Landing Base"
    
#     return "Unknown Base"

# def main():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Erro ao acessar a webcam.")
#         return

#     last_detected_base = "Unknown Base"
#     detected_base = "Unknown Base"
#     display_counter = 0
#     display_threshold = 30  # Número de frames para manter a base detectada na tela
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Erro ao capturar o frame.")
#             break
        
#         edges = preprocess_image(frame)
#         contours = find_contours(edges)
#         current_base_type = classify_base(frame, contours)
        
#         if current_base_type != "Unknown Base":
#             detected_base = current_base_type
#             display_counter = display_threshold
#         elif display_counter > 0:
#             display_counter -= 1
#         else:
#             detected_base = "Unknown Base"
        
#         # Mostrar resultado na tela
#         cv2.putText(frame, f"Detected base: {detected_base}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow("Frame", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Executar a função principal
# main()
#########################################3

import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_circle(image, contour):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return None

def classify_base(image, contours):
    detected_base = "Unknown Base"
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4:  # Verificando se é um retângulo/quadrado
            _, __, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            
            if (1.3 < aspect_ratio < 1.7 and area > 30000):  # Base de Takeoff
                circles = detect_circle(image, approx)
                if circles is not None:
                    detected_base = "Takeoff Base"
                    break
            elif (0.9 < aspect_ratio < 1.1 and area > 9000):  # Base de Pouso
                circles = detect_circle(image, approx)
                if circles is not None:
                    detected_base = "Landing Base"
                    break
    
    return detected_base

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    last_detected_base = "Unknown Base"
    display_counter = 0
    display_threshold = 30  # Número de frames para manter a base detectada na tela
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break
        
        edges = preprocess_image(frame)
        contours = find_contours(edges)
        current_base_type = classify_base(frame, contours)
        
        if current_base_type != "Unknown Base":
            last_detected_base = current_base_type
            display_counter = display_threshold
        elif display_counter > 0:
            display_counter -= 1
        else:
            last_detected_base = "Unknown Base"
        
        # Mostrar resultado na tela
        cv2.putText(frame, f"Detected base: {last_detected_base}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Executar a função principal
main()
