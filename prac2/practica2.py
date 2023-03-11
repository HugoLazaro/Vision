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
blur = cv2.GaussianBlur(gray, (5, 5), 0) # PREGUNTAR: Sigma variable???
show_image(blur, 'Blur')

# Calcular los gradientes de la imagen utilizando el operador de Sobel
# compute the approximate derivatives in the x and y directions
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

show_image(cv2.normalize(sobelx / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient X')
show_image(cv2.normalize(sobely / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient Y')

# Calcular la magnitud y dirección del gradiente
mag = np.sqrt(sobelx**2 + sobely**2)#The magnitude of the gradient is computed using the formula sqrt(Gx^2 + Gy^2)
mag = np.uint8(mag / np.max(mag) * 255) # normalizar la magnitud a valores entre 0 y 255
theta = np.arctan2(sobely, sobelx) + np.pi # PREGUNTAR: En radianes? Rango 0,2pi?

show_image(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Magnitude')
show_image(cv2.normalize(theta/np.pi*128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Theta')

# Aplicar la supresión de no máximos para afinar los bordes detectados
#selecting the local maximum of the gradient magnitude in the direction of the gradient orientation.

# The gradient direction is rounded to one of four possible angles (0, 45, 90, 135). 
# to simplify the subsequent comparisons of gradient magnitude between neighboring pixels.
theta = theta * 135/(2*np.pi) # Rango: [0,135]?
#theta = theta * 180/(np.pi) # Rango: [0,360]?

#-45 a 135 grados
# PREGUNTAR: Por que es la perpendicular???
theta = np.round(theta / 45) * 45
suppressed = np.zeros_like(mag)# store the results of the non-maximum suppression operation
for i in range(1, mag.shape[0]-1):
    for j in range(1, mag.shape[1]-1):
        direction = theta[i, j]
        if direction == 0:
            #abajo, arriba???
            neighbors = [mag[i, j-1], mag[i, j+1]]
        elif direction == 45:
            #arribaIZQ, abajoDER?????
            neighbors = [mag[i-1, j+1], mag[i+1, j-1]]
        elif direction == 90:
            #IZQ, DER??
            neighbors = [mag[i-1, j], mag[i+1, j]]
        else:
            #abajoIZQ, arribaDER??
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

# Itera sobre las coordenadas de los píxeles débiles
for i, j in zip(weak_i, weak_j):
    # Define una ventana de 3x3 alrededor del píxel actual
    i_start = max(0, i-1)
    i_end = min(edges.shape[0], i+2)
    j_start = max(0, j-1)
    j_end = min(edges.shape[1], j+2)
    window = edges[i_start:i_end, j_start:j_end]
    
    # Verifica si algún píxel en la ventana tiene un valor de 255 (es decir, está en los bordes fuertes)
    if np.max(window) == 255:
        # Si sí, establece el valor del píxel actual en 255
        edges[i, j] = 255

# Mostrar la imagen con los bordes detectados
show_image(edges, 'Edges')



def vote_line(i,j,p,theta):
    pass

def hough(img, mag, orientation, threshold):
    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]):
            if mag[i,j]>=threshold:
                x = j - img.shape[0]/2
                y = img.shape[1]/2 - i
                theta = orientation[i,j]
                p = x*np.cos(theta) + y*np.sin(theta)
                vote_line(i,j,p,theta)
     


