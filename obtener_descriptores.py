#A continuación se muestra el código que se uso para obtener los mapas HSV
#y HSL. También se muestra el código para obtener el Histogram of oriented
#gradients, Histograma para Local binary patterns y un histograma de la 
#imagen en escala de gris. Sin embargo, aunque no esta documentado, 
#se obtuvo en pruebas  que su precisión no superaba el 90%. Debido a esto
#Se enfoco los modelos a las Redes Neuronales

#Se importa las librerías a utilizar
import os
import cv2
import numpy as np
####################################################
#Este es el codigo para obtener el Histogram of oriented gradients
def hog(img):
    rows,cols=img.shape
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    ang=ang*180/np.pi
    hog=[0,0,0,0,0,0,0,0]
    for j in range (0,rows):
        for k in range (0,cols):
            if (ang[j,k]==0):
                hog[0]=hog[0]+mag[j,k]
            elif ((ang[j,k]>0) and (ang[j,k]<45)):  
                hog[0]=hog[0]+((45-ang[j,k])/45)*mag[j,k]
                hog[1]=hog[1]+((ang[j,k]-0)/45)*mag[j,k]
            elif (ang[j,k]==45):
                hog[1]=hog[1]+mag[j,k]
            elif ((ang[j,k]>45) and (ang[j,k]<90)):  
                hog[1]=hog[1]+((90-ang[j,k])/45)*mag[j,k]
                hog[2]=hog[2]+((ang[j,k]-45)/45)*mag[j,k]
            elif (ang[j,k]==90):
                hog[2]=hog[2]+mag[j,k]
            elif ((ang[j,k]>90) and (ang[j,k]<135)):  
                hog[2]=hog[2]+((135-ang[j,k])/45)*mag[j,k]
                hog[3]=hog[3]+((ang[j,k]-90)/45)*mag[j,k]
            elif (ang[j,k]==135):
                hog[3]=hog[3]+mag[j,k]
            elif ((ang[j,k]>135) and (ang[j,k]<180)):  
                hog[3]=hog[3]+((180-ang[j,k])/45)*mag[j,k]
                hog[4]=hog[4]+((ang[j,k]-135)/45)*mag[j,k]
            elif (ang[j,k]==180):
                hog[4]=hog[4]+mag[j,k]
            elif ((ang[j,k]>180) and (ang[j,k]<225)):  
                hog[4]=hog[4]+((225-ang[j,k])/45)*mag[j,k]
                hog[5]=hog[5]+((ang[j,k]-180)/45)*mag[j,k]
            elif (ang[j,k]==225):
                hog[5]=hog[5]+mag[j,k]
            elif ((ang[j,k]>225) and (ang[j,k]<270)):  
                hog[5]=hog[5]+((270-ang[j,k])/45)*mag[j,k]
                hog[6]=hog[6]+((ang[j,k]-225)/45)*mag[j,k]
            elif (ang[j,k]==270):
                hog[6]=hog[6]+mag[j,k]
            elif ((ang[j,k]>270) and (ang[j,k]<315)):
                hog[6]=hog[6]+((315-ang[j,k])/45)*mag[j,k]
                hog[7]=hog[7]+((ang[j,k]-270)/45)*mag[j,k]
            elif (ang[j,k]==315):
                hog[7]=hog[7]+mag[j,k]
            elif ((ang[j,k]>315) and (ang[j,k]<361)):  
                hog[7]=hog[7]+((361-ang[j,k])/45)*mag[j,k]
                hog[0]=hog[0]+((ang[j,k]-315)/45)*mag[j,k]
            else:
                print('error de angulo y magnitud')
    hog=hog/np.linalg.norm(hog)
    return hog
######################################
#Este el código para obtener un histograma de Local Binnary patterns
def getlbp(img):
    rows,cols=img.shape
    lbp=np.zeros((rows,cols),dtype=np.uint8)
    for i in range (1,rows-1):
        for j in range (1,cols-1):
            a=img[i,j]
            if a<img[i-1,j-1]:
                lbp[i,j]=lbp[i,j]+127
            if a<img[i,j-1]:
                lbp[i,j]=lbp[i,j]+0
            if a<img[i+1,j-1]:
                lbp[i,j]=lbp[i,j]+2
            if a<img[i-1,j]:
                lbp[i,j]=lbp[i,j]+64
            if a<img[i+1,j]:
                lbp[i,j]=lbp[i,j]+4
            if a<img[i-1,j+1]:
                lbp[i,j]=lbp[i,j]+32
            if a<img[i,j+1]:
                lbp[i,j]=lbp[i,j]+16
            if a<img[i+1,j+1]:
                lbp[i,j]=lbp[i,j]+8
    hist=cv2.calcHist([lbp],[0],None,[256],[0,256])
    hist=hist/np.linalg.norm(hist)
    return hist
#############################################
#Este es el código para obtener el mapa de color HSV
def getmapahsv(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [45, 64], [0, 180, 0, 256] )
    hist = hist/np.linalg.norm(hist)
    return hist
#################################################  
#Este es el código para obtener el mapa de color HSL 
def getmapahsl(img):
    hsl = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    hist = cv2.calcHist( [hsl], [0, 2], None, [45, 64], [0, 180, 0, 256] )
    hist = hist/np.linalg.norm(hist)
    return hist
#####################################################
#Este es el código para obtener el histograma  de color HSL 
def gethslhuehist(img):
    hsl = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    hist = cv2.calcHist( [hsl], [0], None, [180], [0, 180] )
    hist=hist/np.linalg.norm(hist)
    return hist
####################################################
#Este es el código para obtener el histograma de la imagen de escala de grises
def getgrayhist(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist( [gray], [0], None, [256], [0, 256] )
    hist = hist/np.linalg.norm(hist)
    return hist
####################################################

raw_data_dir='Data/Raw/'
destiny_data_dir='Data/'
#Ahora se muestra el código que usa las funciones anteriores para genera la Data
#para que Keras lo use.
for dir1 in ['train/','validation/']:
    for dir2 in['negativo/','positivo/']:
     db1=[]
     db2=[]
     for filename in os.listdir(raw_data_dir+dir1+dir2):
        path_x=raw_data_dir+dir1+dir2+str(filename)
        img=cv2.imread(path_x,1)
        mapahsl=getmapahsl(img)
        mapahsv=getmapahsv(img)
        db1.append(mapahsl)
        db2.append(mapahsv)
     np.save(destiny_data_dir+'mapahsv/'+dir1+dir2+'db.npy',db2)
     np.save(destiny_data_dir+'mapahsl/'+dir1+dir2+'db.npy',db1)       
        
        
        