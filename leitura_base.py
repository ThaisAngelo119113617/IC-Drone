
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
