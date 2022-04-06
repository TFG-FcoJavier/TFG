"""
COMPONENTES PARA UNA RED POR TRANSFERENCIA
"""
import tensorflow as tf

from keras import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.applications import mobilenet

def build_transf_encoder(dim_latente:int, img_shape:tuple, trainable=True) -> Model:
    """
    Éste ``encoder`` va a ser un modelo basado en ``transfer learning``, vamos a tomar la red de ``mobilenet``, entranada para imagenes de ``imagenet`` sin las capa de clasificacion final, con una entrada de tamaño _img\_shape_ y en el output colocamos la 'representacion latente' una codificacion de la imagen que nos permitiría reconstruirla con un ``decoder``.
    """
    inputs = Input(shape=img_shape)
    x=tf.cast(inputs, tf.float32)
    x=tf.keras.layers.Resizing(128,128)(x)
    #x=keras.applications.mobilenet.preprocess_input(x)  #dataformat por defecto es chanel last
    core = mobilenet.MobileNet(input_shape=((128,128,3)), weights="imagenet", include_top=False)
    core.trainable = trainable
    model = core(x, training=trainable)
    model = GlobalAveragePooling2D()(model)
    repr_latente = Dense(dim_latente)(model)
    return Model(inputs, repr_latente)