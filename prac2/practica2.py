import cv2
import numpy as np

def show_image(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Leer la imagen del pasillo
img = cv2.imread('img/Contornos/poster.pgm')#(512, 512, 3)
show_image(img, 'Image')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de suavizado para reducir el ruido de la imagen
blur = cv2.GaussianBlur(gray, (5, 5), 0)
show_image(blur, 'Blur')

# Calcular los gradientes de la imagen utilizando el operador de Sobel
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

show_image(cv2.normalize(sobelx / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient X')
show_image(cv2.normalize(sobely / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient Y')

# Calcular la magnitud y dirección del gradiente
mag = np.sqrt(sobelx**2 + sobely**2)
mag = np.uint8(mag / np.max(mag) * 255) # normalizar la magnitud a valores entre 0 y 255
theta = np.arctan2(sobely, sobelx) * 180 / np.pi

show_image(mag, 'Magnitude')
show_image(cv2.normalize(theta/np.pi*128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Theta')

# Aplicar la supresión de no máximos para afinar los bordes detectados
theta[theta < -45] += 180
theta[theta > 135] -= 180
theta = np.round(theta / 45) * 45
suppressed = np.zeros_like(mag)
for i in range(1, mag.shape[0]-1):
    for j in range(1, mag.shape[1]-1):
        direction = theta[i, j]
        if direction == 0:
            neighbors = [mag[i, j-1], mag[i, j+1]]
        elif direction == 45:
            neighbors = [mag[i-1, j+1], mag[i+1, j-1]]
        elif direction == 90:
            neighbors = [mag[i-1, j], mag[i+1, j]]
        else:
            neighbors = [mag[i-1, j-1], mag[i+1, j+1]]
        if mag[i, j] >= neighbors[0] and mag[i, j] >= neighbors[1]:
            suppressed[i, j] = mag[i, j]


# Aplicar el umbral doble para detectar los bordes finales
low_thresh = 30
high_thresh = 60

edges = np.zeros_like(suppressed)
strong_i, strong_j = np.where(suppressed >= high_thresh)
weak_i, weak_j = np.where((suppressed >= low_thresh) & (suppressed < high_thresh))
edges[strong_i, strong_j] = 255
for i, j in zip(weak_i, weak_j):
    if np.max(edges[max(0, i-1):min(edges.shape[0], i+2), max(0, j-1):min(edges.shape[1], j+2)]) == 255:
        edges[i, j] = 255

# Mostrar la imagen con los bordes detectados
show_image(edges, 'Edges')

# # Aplicar la transformada de Hough para encontrar las líneas en la imagen
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)


# # Dibujar las líneas detectadas en la imagen original
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# # Mostrar la imagen con las líneas detectadas
# cv2.imshow('Lines', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()