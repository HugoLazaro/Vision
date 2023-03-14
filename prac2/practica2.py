import cv2
import numpy as np

def show_image(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################

# La función hough toma como entrada la imagen img, una matriz de magnitudes de gradientes mag, 
# una matriz de orientaciones de gradientes orientation y un umbral threshold para determinar 
# qué píxeles deben ser considerados para la detección de líneas. La función devuelve una matriz 
# accumulator que almacena los votos para cada línea en el espacio de parámetros.

# La función vote_line es llamada por hough para votar por cada línea detectada en la imagen. 
# La función toma como entrada los índices i y j del píxel en la imagen, la distancia p y el 
# ángulo theta de la línea en el espacio de parámetros, y la matriz accumulator que almacena 
# los votos para cada línea en el espacio de parámetros. La función redondea p y theta a 
# enteros y luego incrementa el contador para la línea correspondiente en accumulator.


# def hough(img, mag, orientation, threshold):
#     # Definir la resolución del espacio de parámetros
#     p_resolution = int(np.sqrt((img.shape[0]/2)**2 + (img.shape[1]/2)**2))
#     theta_resolution = 360

#     # Inicializar el acumulador con ceros
#     accumulator = np.zeros((p_resolution, theta_resolution), dtype=np.uint64)

#     # Iterar sobre todos los píxeles de la imagen
#     for i in range(0, img.shape[1]):
#         for j in range(0, img.shape[0]):
#             if mag[i, j] >= threshold:
#                 # Calcular los parámetros de la línea que pasa por el píxel
#                 x = j - img.shape[0]/2
#                 y = img.shape[1]/2 - i
#                 theta = orientation[i, j]
#                 p = x*np.cos(theta) + y*np.sin(theta)

#                 # Votar por la línea en el espacio de parámetros
#                 vote_line(i, j, p, theta, accumulator)

#     return accumulator

# def vote_line(i, j, p, theta, accumulator):
#     """
#     Incrementa el contador para la línea en el espacio de parámetros.
    
#     :param i: índice y del píxel en la imagen.
#     :param j: índice x del píxel en la imagen.
#     :param p: distancia de la línea al origen en el espacio de parámetros.
#     :param theta: ángulo de la línea en el espacio de parámetros.
#     :param accumulator: matriz que almacena los votos para cada línea en el espacio de parámetros.
#     """
#     # Redondea los valores de p y theta a enteros
#     p = int(round(p))
#     theta = int(round(theta))

#     # Calcula la distancia del píxel al centro de la imagen
#     x0, y0 = accumulator.shape[1] // 2, accumulator.shape[0] // 2
#     r = np.sqrt((i - y0)**2 + (j - x0)**2)

#     # Incrementa el contador para la línea en el espacio de parámetros
#     accumulator[int(p+r), theta] += 1

# ###############################################

def hough(img, mag, orientation, threshold):
    rows, cols = img.shape
    accumulator = np.zeros((rows, cols),dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if mag[i, j] >= threshold:
                x = j - cols / 2
                y = rows / 2 - i
                theta = orientation[i, j]
                p = x * np.cos(theta) + y * np.sin(theta)
                vote_line(accumulator, p, theta)

    return accumulator

def vote_line(accumulator, p, theta, rho_res=1, theta_res=np.pi/180):
    rows, cols = accumulator.shape
    rho_max = np.sqrt(rows**2 + cols**2)

    rho = int((p + rho_max) / rho_res)
    t = int(theta / theta_res)

    if rho >= 0 and rho < rows and t >= 0 and t < cols:
        accumulator[rho, t] += 1

def vote_line(i, j, p, theta, accumulator):
    a = math.cos(theta)
    b = math.sin(theta)
    x = int(p * a + j)
    y = int(p * b + i)
    if x >= 0 and x < accumulator.shape[1] and y >= 0 and y < accumulator.shape[0]:
        accumulator[y, x] += 1

###############################################



# Leer la imagen del pasillo
img = cv2.imread('img/Contornos/poster.pgm')#(512, 512, 3)
show_image(img, 'Image')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de suavizado para reducir el ruido de la imagen
sigma = 1
kernel_size = 5*sigma


blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma) # PREGUNTAR: Sigma variable???
#show_image(blur, 'Blur')
sigma
# Calcular los gradientes de la imagen utilizando el operador de Sobel
# compute the approximate derivatives in the x and y directions
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

#show_image(cv2.normalize(sobelx / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient X')
#show_image(cv2.normalize(sobely / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient Y')

# Calcular la magnitud y dirección del gradiente
mag = np.sqrt(sobelx**2 + sobely**2)#The magnitude of the gradient is computed using the formula sqrt(Gx^2 + Gy^2)
mag = np.uint8(mag / np.max(mag) * 255) # normalizar la magnitud a valores entre 0 y 255
theta = np.arctan2(sobely,sobelx) + np.pi # PREGUNTAR: En radianes? Rango 0,2pi?

show_image(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Magnitude')
#show_image(cv2.normalize(theta/np.pi*128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Theta')

# Aplicar la supresión de no máximos para afinar los bordes detectados
#selecting the local maximum of the gradient magnitude in the direction of the gradient orientation.

# The gradient direction is rounded to one of four possible angles (0, 45, 90, 135). 
# to simplify the subsequent comparisons of gradient magnitude between neighboring pixels.
theta = theta * 180/(np.pi) # Rango: [0,135]
#theta = theta * 180/(np.pi) # Rango: [0,360]

theta = np.round((theta) / 22.5)
#theta = (theta / 22.5)
suppressed = np.zeros_like(mag)# store the results of the non-maximum suppression operation
for i in range(1, mag.shape[0]-1):
    for j in range(1, mag.shape[1]-1):
        (direction) = int(theta[i, j])
        if direction == 0 or direction == 16 or direction == 15 or direction == 7 or direction == 8:
            #abajo, arriba
            neighbors = [mag[i, j-1], mag[i, j+1]]
        elif direction == 13 or direction == 14 or direction == 5 or direction == 6:
            #arribaIZQ, abajoDER
            neighbors = [mag[i-1, j+1], mag[i+1, j-1]]
        elif direction == 11 or direction == 12 or direction == 3 or direction == 4:
            #IZQ, DER
            neighbors = [mag[i-1, j], mag[i+1, j]]
        elif direction == 9 or direction == 10 or direction == 1 or direction == 2:
            #abajoIZQ, arribaDER
            neighbors = [mag[i-1, j-1], mag[i+1, j+1]]

        if mag[i, j] >= neighbors[0] and mag[i, j] >= neighbors[1]:
            suppressed[i, j] = mag[i, j]

strong_i, strong_j = np.where(suppressed > 20)
suppressed[strong_i,strong_j] = 255
#show_image(cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Supressed')




# def vote_line(i,j,p,theta):
#     pass

# def hough(img, mag, orientation, threshold):
#     for i in range(0, img.shape[1]):
#         for j in range(0, img.shape[0]):
#             if mag[i,j]>=threshold:
#                 x = j - img.shape[0]/2
#                 y = img.shape[1]/2 - i
#                 theta = orientation[i,j]
#                 p = x*np.cos(theta) + y*np.sin(theta)
#                 vote_line(i,j,p,theta)
