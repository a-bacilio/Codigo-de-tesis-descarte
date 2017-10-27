#Este programa muestra el codigo para entrenar una red neuronal basado en el
#bloque Resnet

#Para empezar se carga las librerias necesarias de Keras y de Numpy
from keras.models import Model
from keras.models import Sequential
from keras.layers import merge
from keras.layers import normalization
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers import Convolution2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint,Callback,ReduceLROnPlateau
import numpy as np

#Ya que el código es casi el mismo para los 8 entrenamientos (solo varia el modelo
# y/o la base de datos) se usa un nombre código
codename='mapahsl_resnet'
data_dir='Data/mapahsl/'


#Esto declara el uso de un registro del learning rate, el cual es útil para ver
#cuantas veces se estanca el entrenamiento y si el entrenamiento avanza o no.
registro_lr=[]
class showlr(Callback):
    def on_train_begin(self, logs={}):
        registro_lr = np.load(codename +'.npy',[])
    def on_epoch_end(self, batch, logs={}):
        lr = self.model.optimizer.lr.get_value()
        registro_lr.append(lr)
        np.save(codename + '.npy',registro_lr)
        print(lr)
        




# Se carga la data necesaria para el entrenamiento (en este caso, la base de
#datos de los mapas HSL obtenidos de las imagenes de cacao sano y enfermos)
#del sistema y se les da una forma aceptable para Keras

#Los datos para el entreamiento
train1 = np.load(data_dir+'train/negativo/db.npy').tolist()
train=train1.copy()
train2 = np.load(data_dir+'train/positivo/db.npy').tolist()
for i in range(len(train2)):
    train.append(train2[i])
train=np.array(train)
train=train.reshape(train.shape[0],1, 45, 64)
train_labels=np.array([0]*len(train1)+[1]*len(train2))
train_labels=train_labels.reshape(train_labels.shape[0],1)

#los datos para la validacion
val1 = np.load(data_dir+'validation/negativo/db.npy').tolist()
val=val1.copy()
val2 = np.load(data_dir+'validation/positivo/db.npy').tolist()
for i in range(len(val2)):
    val.append(val2[i])
val=np.array(val)
val=val.reshape(val.shape[0],1,45, 64)
val_labels=np.array([0]*len(val1)+[1]*len(val2))
val_labels=val_labels.reshape(val_labels.shape[0],1)
img_height,img_width = train.shape[2],train.shape[3]

#Esta variable indica el numero de elementos a buscar, sin embargo al colocar 
# 1 se vuelve una clasificacion binaria. donde 1 es un elemento de una clase y
# 0 del otro. En este caso, positivo (infectado) tiene valor 1
nb_classes = 1

#A continuacion, se muestra el código del módelo a entrenar el cual usa 
#el bloque Resnet
#####################################################################
#Se inicia el modelo con Sequential
model = Sequential()
#Se prepara la entrada para una imagen de 1 solo canal de 45x 64 pixeles
#el cual es el mapa de color HSL
Input_0=Input(shape=(1,45,64))

#Se inicia usando una serie de 64 filtros de 7x7 los cuales convolucionan
#con la imagen para que puedan obtener caracteristicas de los mapas de color de
#la imagen de cacao
tronco_conv1=Convolution2D(64, 7, 7,
                           subsample=(2, 2),
                           border_mode='same')(Input_0)

#Se reduce con un Subsampling de 2x2 para reducir el tamaño de la imagen 
#convolucionada
tronco_pool1=MaxPooling2D(pool_size=(2, 2),
                            strides=[2,2],
                            border_mode='valid',
                            dim_ordering='default')(tronco_conv1)

#Ahora se iniciara el bloque Resnet
rama1_res_BN1=normalization.BatchNormalization(mode=0,
                                               axis=-1)(tronco_pool1)

rama1_res_activation1=Activation('relu')(rama1_res_BN1)

rama1_dp1=Dropout(0.5)(rama1_res_activation1)

rama1_res_conv1=Convolution2D(64, 3, 3,
                           subsample=(1, 1),
                           border_mode='same')(rama1_dp1)

rama1_res_BN2=normalization.BatchNormalization(mode=0,
                                               axis=-1)(rama1_res_conv1)

rama1_res_activation2=Activation('relu')(rama1_res_BN2)

rama1_dp2=Dropout(0.5)(rama1_res_activation2)

rama1_res_conv2=Convolution2D(64, 3, 3,
                           subsample=(1, 1),
                           border_mode='same')(rama1_dp2)

#Se termina el bloque de resnet sumando el inicio del bloque con el 
#final del bloque
reconexion_res_tronco=merge([rama1_res_conv2,tronco_pool1], mode="sum")

#Se hace un subsampling promediando de matrices de 2x2 en matrices de 2x2
tronco_pool2=AveragePooling2D(pool_size=(4, 4),
                 strides=[4,4],
                 border_mode='valid',
                 dim_ordering='default')(reconexion_res_tronco)
#*Todos los dropouts apagan neuronas momentaneamente para
#que el sistema aprenda de maneras diferentes y no de una sola manera, 
#sino podria tender a memorizar los elementos de entreamiento envés de 
#aprender a diferenciarlos
tronco_dp1=Dropout(0.5)(tronco_pool2)

tronco_flat=Flatten()(tronco_dp1)
#Se reduce todo a un solo valor que indique si es o no un cacao infectado
tronco_fc1=Dense(1,activation="sigmoid")(tronco_flat)


model=Model(input=Input_0,output=tronco_fc1)
#Se compila en el sistema Keras usando Theano o Tensorflow, ademas se define el
#elemento a minimizar para la optimización y las metricas a mostrar(precisión,
#loss,etc)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#######################################################################

#Ahora agregamos al entrenamiento caracteristicas útiles en el caso de
#accidentes y para realizar el registro del entrenamiento

#Esta es la direccion donde se guarda un modelo despues de entrenar cada ciclo
#y la funcion que la guarda
direccion='modelos/'+codename+'/weights.{epoch:04d}-{acc:.4f}-{loss:.4f}-{val_acc:.4f}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(direccion,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max')

#Esta caracteristica reduce el learning rate cada vez que se el loss no disminuye
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,patience=1, verbose=1,  cooldown=0, min_lr=0)
#Esta caracteristica llama a la función explicada anteriormente que almacena el valor del
#learning rate
histlr=showlr()

#Esta caracteristica llma a las anteriores para agregarlos al entrenamiento
callbacks_list = [checkpoint,reduce_lr,histlr]

#Ahora se entrena el sistema. batch_size define el tamaño de  la porción de 
#que va a memoria ram para el entrenamiento. Asi va de porción en porción hasta 
#acabar un ciclo. nb_epoch, define el numero de ciclos de entrenamiento
#callbacks llama a las características adicionales
#verbose sirve para mostrar el proceso del entrenamiento en  el prompt de la linea
#de comandos.
#val y train es la data que se usa para la validacion y el entrenamiento
#respectivamente.
model.fit(train, train_labels,
          batch_size=8,
          nb_epoch=1000,
          callbacks=callbacks_list,shuffle=True,
          verbose=1, validation_data=(val, val_labels))