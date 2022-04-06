"""
COMPONENTES PARA UNA RED DENSA
"""
import keras
import numpy as np

from keras import Model
from keras.layers import Dense, LeakyReLU, Flatten, Reshape
from misc import pyramid_rule


def build_dense_encoder_P(dim_latente:int, img_shape:tuple, depth=2) -> Model:
    """
    Encoder denso, sigue la regla de la piramide.\n
    dim_latente: tamaño del output\n
    img_shape: forma de la imagen para la entrada\n
    depth: numero de capas ocultas
    """
    layer_sizes = pyramid_rule(depth, np.prod(img_shape), dim_latente)
    model =  keras.Sequential()
    model.add(Flatten(input_shape=img_shape))
    for ls in layer_sizes:
        model.add(Dense(ls))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(dim_latente))
    return model

def build_dense_decoder_P(dim_latente:int, img_shape:tuple, depth=2) -> Model:
    """
    Decoder denso, sigue la regla de la piramide.\n
    dim_latente: tamaño del input\n
    img_shape: forma de la imagen para la salida\n
    depth: numero de capas ocultas
    """
    model = keras.Sequential()
    layer_sizes = pyramid_rule(depth, dim_latente, np.prod(img_shape))
    model.add(Dense(layer_sizes[0], input_dim=dim_latente))
    model.add(LeakyReLU(alpha=0.2))
    for i in range(1, depth):
        model.add(Dense(layer_sizes[i], input_dim=dim_latente))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation=keras.activations.sigmoid))
    model.add(Reshape(img_shape))
    
    return model

def build_dense_encoder(dim_latente:int, img_shape:tuple, depth=2, width=1000) -> Model:
    """
    Encoder denso.\n
    dim_latente: tamaño del output\n
    img_shape: forma de la imagen para la entrada\n
    depth: numero de capas ocultas\n
    width: numero de neuronas de las capas ocultas
    """
    layer_sizes=list(width for _ in range(depth))
    model =  keras.Sequential()
    model.add(Flatten(input_shape=img_shape))
    for ls in layer_sizes:
        model.add(Dense(ls))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(dim_latente))
    return model

def build_dense_decoder(dim_latente:int, img_shape:tuple, depth=2, width=1000) -> Model:
    """
    Decoder denso.\n
    dim_latente: tamaño del output\n
    img_shape: forma de la imagen para la entrada\n
    depth: numero de capas ocultas\n
    width: numero de neuronas de las capas ocultas
    """
    model = keras.Sequential()
    layer_sizes=list(width for _ in range(depth))
    model.add(Dense(layer_sizes[0], input_dim=dim_latente))
    model.add(LeakyReLU(alpha=0.2))
    for i in range(1, depth):
        model.add(Dense(layer_sizes[i], input_dim=dim_latente))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation=keras.activations.sigmoid))
    model.add(Reshape(img_shape))
    
    return model