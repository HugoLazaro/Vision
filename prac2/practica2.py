import cv2
import numpy as np

def show_image(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def non_maximal_suppression():
    pass

# Normalizr angle from -pi to pi to 0 to 2pi
def normalize_angle(angle):
    if angle < 0:
        return 2*np.pi + angle
    else:
        return angle

# Angle from 0 to 2pi to degrees
def angle_to_degrees(angle):
    return angle * 180 / np.pi

def hough_lines(gray_hough_lines):

    dst = cv2.Canny(gray_hough_lines, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)


# Hough transform
def hough(img, mag, m):
    img_save = img.copy()
    votes = np.zeros(img.shape[1])
    #height_votes = int(img.shape[0]/2) -40 #Pasillo 1
    height_votes = int(img.shape[0]/2) #Pasillo 2y3
    print(height_votes)
    threshold = 10

    # # Encontramos los índices de los valores mayores que 5
    # indices = np.where(m < 0)

    # # Imprimimos los índices encontrados
    # print(indices)
    print(mag.shape)


    for j in range(0, mag.shape[1]):#Recorre columnas(horiontal) -> x 
        for i in range(0, mag.shape[0]):#Recorre filas(vertical) -> y
            if mag[i,j]>=threshold:
                direction = float(m[i, j]) 
                direction = angle_to_degrees(direction)
                #if (direction < 10 or direction > 350) or (direction > 170 and direction < 190):  verticales  
                #if (direction < 100 and direction > 80) or (direction > 260 and direction < 280): horizontales
                if not((direction < 10 or direction > 350) or (direction > 170 and direction < 190)
                    or (direction < 100 and direction > 80) or (direction > 260 and direction < 280)):
                    b = j - (m[i,j])*i
                    y = height_votes
                    # x = (y - b) / np.tan(m[i,j])
                    x = (y - b) / (m[i,j])
                    
                    # cv2.circle(img, (int(j), int(i)), 1, (255, 255, 0), -1)
                

                    x = int(x)
                    y = int(y)
                    if x < img_save.shape[1]-2 and x > 1:
                        
                        cv2.circle(img_save, (j, i), 1, (255, 255, 0), -1)

                        
                        #if x > 100:
                            #cv2.circle(img_save, (int(j), int(i)), 1, (255, 255, 0), -1)
                            #cv2.line(img_save, (j,i), (x,y), (255,255,0), 1, cv2.LINE_AA)
                        votes[x] = votes[x] + 3
                        votes[x+1] = votes[x+1] + 2
                        votes[x+2] = votes[x+2] + 1
                        votes[x-1] = votes[x-1] + 2
                        votes[x-2] = votes[x-2] + 1
                        #cv2.circle(img_save, (int(x), int(y)), 1, (255, 255, 0), -1)
                        
    cv2.imshow('Imagen con punto de fuga', img_save)
    cv2.waitKey()

    #print(votes)

    vanishing_point = np.argmax(votes)
    return vanishing_point





# Leer la imagen del pasillo
img = cv2.imread('img/Contornos/pasillo3.pgm')#(512, 512, 3)
show_image(img, 'Image')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de suavizado para reducir el ruido de la imagen
sigma = 1
kernel_size = 5*sigma


blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma) # PREGUNTAR: Sigma variable???
#show_image(blur, 'Blur')

# Calcular los gradientes de la imagen utilizando el operador de Sobel
# compute the approximate derivatives in the x and y directions
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

#show_image(cv2.normalize(sobelx / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient X')
#show_image(cv2.normalize(sobely / 2 + 128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Gradient Y')

# Calcular la magnitud y dirección del gradiente
mag = np.sqrt(sobelx**2 + sobely**2)#The magnitude of the gradient is computed using the formula sqrt(Gx^2 + Gy^2)
mag = np.uint8(mag / np.max(mag) * 255) # normalizar la magnitud a valores entre 0 y 255
mag_save = mag.copy()
theta = np.arctan2(sobely,sobelx) # PREGUNTAR: En radianes? Rango 0,2pi?


# Creamos una versión vectorizada de la función
normalize_angle_vectorized = np.vectorize(normalize_angle)

# Aplicamos la función vectorizada a cada elemento de la matriz
theta = normalize_angle_vectorized(theta)
m = theta.copy()

#show_image(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Magnitude')
#show_image(cv2.normalize(theta/np.pi*128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Theta1')

# Aplicar la supresión de no máximos para afinar los bordes detectados
#selecting the local maximum of the gradient magnitude in the direction of the gradient orientation.

# The gradient direction is rounded to one of four possible angles (0, 45, 90, 135). 
# to simplify the subsequent comparisons of gradient magnitude between neighboring pixels.
# theta = theta * 180/(np.pi) # Rango: [0,135]
# #theta = theta * 180/(np.pi) # Rango: [0,360]



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
# show_image(cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), 'Supressed')



# HOUGH
gray_hough_lines = gray.copy()
vanishing_point = hough(gray, suppressed, m)


hough_lines(gray_hough_lines)


print("van: " + str(vanishing_point))
print("center: " + str(img.shape[0]/2))

#cv2.circle(img, (int(vanishing_point), int(img.shape[0]/2)-40), 5, (0, 0, 255), -1) #Pasillo1
cv2.circle(img, (int(vanishing_point), int(img.shape[0]/2)), 5, (0, 0, 255), -1) #Pasillo2y3

cv2.imshow('Imagen con rectas de Hough', img)
cv2.waitKey()