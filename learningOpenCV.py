import cv2
import numpy as np

def binariza(imagem):

    imgTransformada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(imgTransformada, 200, 255, cv2.THRESH_BINARY)

    return blackAndWhiteImage

def removeRuido(imagem):

    imgTransformada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(imgTransformada, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.imwrite('removednoise.jpg', img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('tresh.jpg', img)

    return img


# carregando imagem em uma variavel
path = '/home/matheuswmb/PycharmProjects/procImagem/texto.png'
img = cv2.imread(path)

# mostrando imagem
#cv2.imshow('image', img)

#imagemBinarizada = binariza(img)
#cv2.imshow('image2', imagemBinarizada)

imagemSemRuido = removeRuido(img)
cv2.imshow('image3', imagemSemRuido)

# aguarda input do usuario para fechar imagem
cv2.waitKey(0)

# fecha todas as janelas abertas
cv2.destroyAllWindows()


