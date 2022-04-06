"""
DISCRIMINADORES ADVERSARIOS

El discriminador va a tener como entrada la codificacion latente de las imagenes y como salida una neurona que discrimina entre imagenes "reales" y "falsas". De esta forma entrenamos al encoder para que codifique con la distribucion que usemos para generar las "imagenes reales", en este caso, una distribución normal.
"""
import tensorflow as tf
import keras

from keras import Model, Sequential
from keras.layers import Dense, LeakyReLU, Input

def build_discriminator(dim_latente:int, depth = 2, width=1000) -> Model:
    """
    Discriminador. Predice si una coordenada en el espacio latente pertenece a una imagen real o no\n
    dim_latente: tamaño del output\n
    depth: numero de capas ocultas\n
    width: numero de neuronas de las capas ocultas
    """
    layer_sizes=list(width for _ in range(depth))
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_dim=dim_latente))
    model.add(LeakyReLU(alpha=0.2))
    for i in range(1, depth):
        model.add(Dense(layer_sizes[i]))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation=keras.activations.sigmoid))
    encoded = Input(shape=dim_latente)
    valid = model(encoded)
    return keras.Model(encoded, valid)

# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def build_class_discriminator(dim_latente, clases, depth = 2, width=1000) -> Model:
    """
    Discriminador de clases. Introduce informacion sobre la etiqueta a la que pertenece la imagen.\n
    Las etiquetas y la coordenada latente se combinan en un input\n
    dim_latente: tamaño del output\nç
    clases: numero de etiquetas posibles para el conjunto de los datos\n
    depth: numero de capas ocultas\n
    width: numero de neuronas de las capas ocultas
    """
    layer_sizes=list(width for _ in range(depth))
    latent_input = Input(shape=dim_latente)
    class_input = Input(shape=clases)
    concated_input = tf.keras.layers.concatenate([latent_input, class_input])
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_dim=dim_latente+clases))
    model.add(LeakyReLU(alpha=0.2))
    for i in range(1, depth):
        model.add(Dense(layer_sizes[i]))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation=keras.activations.sigmoid))
    output = model(concated_input)
    return keras.Model([latent_input, class_input], output)

# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def build_branched_class_discriminator(dim_latente:int, clases:int, depth = 2, width=1000) -> Model:
    """
    Discriminador de clases. Introduce informacion sobre la etiqueta a la que pertenece la imagen.\n
    Las etiquetas y la coordenada latente predicen por separado y se juntan al final en una neurona\n
    dim_latente: tamaño del output\nç
    clases: numero de etiquetas posibles para el conjunto de los datos\n
    depth: numero de capas ocultas\n
    width: numero de neuronas de las capas ocultas
    """
    # Rama latente
    #latent_layer_sizes = pyramid_rule(depth, dim_latente, 1)
    latent_layer_sizes=list(width for _ in range(depth))
    latent_input = Input(shape=dim_latente)
    currentLayer = latent_input
    for i in range(0, depth):
        currentLayer = Dense(latent_layer_sizes[i])(currentLayer)
        currentLayer = LeakyReLU(alpha=0.2)(currentLayer)
    x = keras.Model(inputs=latent_input, outputs=currentLayer)

    # Rama de clases
    #class_layer_sizes = pyramid_rule(depth, clases, 1)
    class_layer_sizes=list(width for _ in range(depth))
    class_input = Input(shape=clases)
    currentLayer = class_input
    for i in range(0, depth):
        currentLayer = Dense(class_layer_sizes[i])(currentLayer)
        currentLayer = LeakyReLU(alpha=0.2)(currentLayer)
    y = keras.Model(inputs=class_input, outputs=currentLayer)

    combined = tf.keras.layers.concatenate([x.output, y.output])
    output = Dense(2)(combined)
    output = Dense(1)(output)
    return keras.Model(inputs = [x.input, y.input], outputs=output)