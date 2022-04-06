"""
COMPONENTES PARA UNA RED CONVOLUCIONAL
"""
import keras
import numpy as np

from keras import Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Conv2DTranspose

def build_conv_encoder(dim_latente:int, img_shape:tuple, depth=2, filter = 32) -> Model:
    """
    Encoder convolucional. Comienza con 8 filtros de (3,3) y cada capa duplica el numero de filtros\n
    dim_latente: tamaño del output\n
    img_shape: forma de la imagen para la entrada\n
    depth: numero de capas ocultas\n
    """
    model =  keras.Sequential()
    model.add(Conv2D(filter, (3,3), padding="same", activation="relu", input_shape=img_shape))
    model.add(MaxPooling2D())
    for _ in range(depth-1):
        filter*=2
        model.add(Conv2D(filter, (3,3), padding="same", activation="relu"))
        model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(dim_latente))
    
    return model

def build_conv_decoder(dim_latente:int, img_shape:tuple, depth=2, filter = 32) -> Model:
    """
    Decoder convolucional. Infiere las convoluciones de las capas asumiendo que la ultima convolucoion tiene 8 filtros de (3,3)\n
    dim_latente: tamaño del input\n
    img_shape: forma de la imagen para la salida\n
    depth: numero de capas ocultas\n
    """
    startLayer = list(int(np.floor(d/(2**depth))) for d in img_shape)
    filter *=(2 ** (depth-1))
    startLayer[-1] = filter
    startLayer = tuple(startLayer) 

    model=keras.Sequential()
    model.add(Dense(dim_latente, input_dim=(dim_latente)))
    model.add(Dense(np.prod(startLayer)))              
    model.add(Reshape(startLayer))      
    for _ in range(depth):
        model.add(Conv2DTranspose(filter, kernel_size=(3,3), strides=2, padding="same", activation="relu"))
        filter/=2
    model.add(Conv2D(img_shape[-1], (3, 3), padding="same", activation=keras.activations.sigmoid))
    
    return model
