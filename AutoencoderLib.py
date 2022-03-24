"""
Este modulo contiene todas las funciones para la creacion, entrenamiento y muestreo de un AAE, incluyendo cada una de las partes y todos los tipos
"""

from math import ceil
import tensorflow as tf
from tensorflow import keras
if __name__=="__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from keras import Sequential, Model
from keras import losses, metrics, optimizers
from keras.applications import mobilenet
from keras.layers import Input, Dense, Conv2D, Flatten, LeakyReLU, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, Reshape

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from timeit import default_timer as timer

from utilities import unpickle, mkfolders

# -----------------------------------------------------------
# FUNCIONES VARIAS
# Cálculo de tamaño de capas, losses personalizadas, etc...
# -----------------------------------------------------------

def pyramid_rule(h_layers:int, input_size:int, output_size:int) -> list:
    """
    Devuelve una lista con el tamaño de las capas para una red neuronal, siguiendo la regla de la piramide geometrica.\n
    h_layers: numero de capas ocultas deseadas\n
    input_size: tamaño de la capa de entrada\n
    output_size: tamaño de la capa de salida
    """
    layers = []
    if h_layers < 1:
        print("No layers")
        return []
    print("Layers for input %d and output %d:" % (input_size,  output_size))
    rate = (input_size/output_size)**(1/(h_layers+1))
    for l in range(h_layers):
        layer_size = output_size*(rate**(h_layers-l))
        layer_size = round(layer_size)
        layers.append(layer_size)
        print("Layer %d: %d neurons" % (l+1, layer_size))
    return layers

def EMD_loss(y_true, y_pred):
    """
    Earth mover's distance, para usar al compilar un modelo
    """
    n = np.prod(y_true.shape)
    p = tf.math.subtract(y_true, y_pred)
    p = tf.math.square(p)
    p = tf.math.reduce_sum(p)
    return tf.math.sqrt(tf.math.divide(p,n))

# -----------------------------------------------------------
# COMPONENTES PARA UNA RED DENSA
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# COMPONENTES PARA UNA RED CONVOLUCIONAL
# -----------------------------------------------------------
def build_conv_encoder(dim_latente:int, img_shape:tuple, depth=2) -> Model:
    """
    Encoder convolucional. Comienza con 8 filtros de (3,3) y cada capa duplica el numero de filtros\n
    dim_latente: tamaño del output\n
    img_shape: forma de la imagen para la entrada\n
    depth: numero de capas ocultas\n
    """
    model =  keras.Sequential()
    filter = 8
    model.add(Conv2D(filter, (3,3), padding="same", activation="relu", input_shape=img_shape))
    model.add(MaxPooling2D())
    for _ in range(depth-1):
        filter*=2
        model.add(Conv2D(filter, (3,3), padding="same", activation="relu"))
        model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(dim_latente))
    
    return model

def build_conv_decoder(dim_latente:int, img_shape:tuple, depth=2) -> Model:
    """
    Decoder convolucional. Infiere las convoluciones de las capas asumiendo que la ultima convolucoion tiene 8 filtros de (3,3)\n
    dim_latente: tamaño del input\n
    img_shape: forma de la imagen para la salida\n
    depth: numero de capas ocultas\n
    """
    startLayer = list(int(np.floor(d/(2**depth))) for d in img_shape)
    filter = 8 * (2 ** (depth-1))
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

# -----------------------------------------------------------
# COMPONENTES PARA UNA RED POR TRANSFERENCIA
# -----------------------------------------------------------

# https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input
# https://keras.io/guides/transfer_learning/
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

# -----------------------------------------------------------
# DISCRIMINADORES ADVERSARIOS
# -----------------------------------------------------------

# El discriminador va a tener como entrada la codificacion latente de las imagenes y como salida una neurona que discrimina entre imagenes "reales" y "falsas". De esta forma entrenamos al encoder para que codifique con la distribucion que usemos para generar las "imagenes reales", en este caso, una distribución normal.
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

# -----------------------------------------------------------
# COMPILADO DE MODELOS
# -----------------------------------------------------------

def assemble_AAE_twoPhased(dim_latente:int, img_shape:tuple, enc_model:Model=build_dense_encoder, dec_model:Model=build_dense_decoder, disc_model:Model = build_discriminator, 
                compilation_kwargs:dict={}, enc_kwargs:dict={}, dec_kwargs:dict={}, disc_kwargs:dict={}) -> tuple:
    """
    Metodo para la construccion. Devuelve una tupla con los modelos del encoder, decoder, discriminador y autoencoder adversario en ese orden.\n
    dim_latente: dimensiones del espacio latente \n
    img_shape: tamaño de la imagen \n
    enc_model: funcion de construccion del encoder \n
    dec_model: funcion de construccion del decoder\n
    disc_model: funcion de construccion del discriminador \n
    compilation_kwargs: agrupacion de argumentos "ae_loss", "disc_loss", "optimizer" para la complilacion del discriminador y el autoencoder \n
    enc_kwargs: argumentos admitidos por el encoder \n
    dec_kwargs: argumentos admitidos por el decoder\n
    disc_kwargs: argumentos admitidos por el discriminador
    """
    #Parameters
    disc_params = {"dim_latente":dim_latente}
    disc_params.update(disc_kwargs)
    enc_params = {"dim_latente":dim_latente, "img_shape":img_shape}
    enc_params.update(enc_kwargs)
    dec_params = {"dim_latente":dim_latente, "img_shape":img_shape}
    dec_params.update(dec_kwargs)

    cp = {"ae_loss":losses.mean_squared_error, 
          "disc_loss": losses.binary_crossentropy, 
          "optimizer" : keras.optimizers.Adam(0.0002, 0.5)
         } #cp = compilation params
    cp.update(compilation_kwargs)
    # Discriminador
    discriminator = disc_model(**disc_params)
    discriminator.compile(loss=cp["disc_loss"], metrics="accuracy")

    # Encoder y decoder
    encoder = enc_model(**enc_params)
    decoder = dec_model(**dec_params)

    # Autoencoder
    # el encoder toma un imagen y la codifica y el decoder toma la codificacion e intenta regenerar la imagen
    img = Input(shape=img_shape, name="data")
    encoded = encoder(img)
    reconstructed = decoder(encoded)
    
    if "clases" in disc_params.keys():
        clase = Input(shape=disc_params["clases"], name = "labels")
        disc_input = [encoded, clase]
        aae_input = [img, clase]
    else:
        disc_input = encoded
        aae_input = img

    # para el autoencoder adversario solo queremos entrenar el generador, no el discriminador
    discriminator.trainable=False

    # El discriminador evalua la validez de la codificacion
    validez = discriminator(disc_input)

    # Autoencoder adversario 
    a_autoencoder = keras.Model(aae_input, [reconstructed, validez])
    a_autoencoder.compile(loss=[cp["ae_loss"], cp["disc_loss"]], optimizer=cp["optimizer"])#, loss_weights=[0.999, 0.001])
    return (encoder, decoder, discriminator, a_autoencoder)

def assemble_AAE_threePhased(
    dim_latente:int, img_shape:tuple, enc_model:Model=build_dense_encoder, dec_model:Model=build_dense_decoder, disc_model:Model = build_discriminator, compilation_kwargs:dict={}, enc_kwargs:dict={}, dec_kwargs:dict={}, disc_kwargs:dict={}
    ) -> tuple:
    """
    Metodo para la construccion. Devuelve una tupla con los modelos del encoder, decoder, discriminador, autoencoder y encoder-discriminador adversario en ese orden.\n
    dim_latente: dimensiones del espacio latente \n
    img_shape: tamaño de la imagen \n
    enc_model: funcion de construccion del encoder \n
    dec_model: funcion de construccion del decoder\n
    disc_model: funcion de construccion del discriminador \n
    compilation_kwargs: agrupacion de argumentos "ae_loss", "disc_loss", "optimizer" para la complilacion del discriminador y el autoencoder \n
    enc_kwargs: argumentos admitidos por el encoder \n
    dec_kwargs: argumentos admitidos por el decoder\n
    disc_kwargs: argumentos admitidos por el discriminador
    """
    # Discriminador
    discriminator = disc_model(dim_latente=dim_latente, **disc_kwargs)
    cp = {"ae_loss":losses.mean_squared_error, 
          "disc_loss": losses.binary_crossentropy, 
          "optimizer" : keras.optimizers.Adam(0.0002, 0.5)
         } #cp = compilation params
    cp.update(compilation_kwargs)
    discriminator.compile(loss=cp["disc_loss"], metrics="accuracy")

    # Encoder y decoder
    encoder = enc_model(dim_latente=dim_latente, img_shape=img_shape, **enc_kwargs)
    decoder = dec_model(dim_latente=dim_latente, img_shape=img_shape,**dec_kwargs)

    # Autoencoder
    # el encoder toma un imagen y la codifica y el decoder toma la codificacion e intenta regenerar la imagen
    img = Input(shape=img_shape, name="data")
    encoded = encoder(img)
    reconstructed = decoder(encoded)
    autoencoder = Model(img, reconstructed)
    autoencoder.compile(loss=cp["ae_loss"], optimizer=cp["optimizer"])

    # para la distribucion del encoder solo queremos entrenar el generador, no el discriminador
    discriminator.trainable=False

    if "clases" in disc_kwargs.keys():
        clase = Input(shape=disc_kwargs["clases"], name = "labels")
        disc_input = [encoded, clase]
        enc_input = [img, clase]
    else:
        disc_input = encoded
        enc_input = img

    # El discriminador evalua la validez de la codificacion
    validez = discriminator(disc_input)

    # Encoder-discriminador adversario 
    encosriminator = keras.Model(enc_input, validez)
    encosriminator.compile(optimizer=cp["optimizer"], loss='binary_crossentropy', metrics=['accuracy'])

    return (encoder, decoder, discriminator, autoencoder, encosriminator)

# -----------------------------------------------------------
# GENERACION DE EJEMPLOS
# -----------------------------------------------------------

def true_sampler(dim_latente:int, batch_size:int) -> np.ndarray:
    """
    Devuelve un array de numpy que contiene ejemplos de coordenadas del espacio latente pertenecientes a una distribucion normal\n
    dim_latente: tamaño de los ejemplos\n
    batch_size: numero de ejemplos
    """
    return np.random.normal(size=(batch_size, dim_latente))

def onehotify(labels:list, nclases:int):
    """
    Revuelve un array de etiquetas [2,0,2,1,0,...] transformada en una lista de one hot vector [[0,0,1],[1,0,0],...]\n
    labels: lista de etiquetas a transformar\n
    nclases: numero de posibles etuquetas (tamaño del vector one hot)
    """
    onehotlabels = []
    for label in labels:
        thisLabel = np.zeros(nclases)
        thisLabel[label]=1
        onehotlabels.append(thisLabel)
    return np.array(onehotlabels)

def true_sampler_clases(dim_latente:int, batch_size:int, nclases:int):
    """
    Devuelve un array de numpy que contiene ejemplos de coordenadas del espacio latente, cada una pertenecientes a una distribucion normal distinta dependiendo de su etiqueta.
    Estas distribuciones se encuentan en la diagonal del espacio latente\n
    dim_latente: tamaño de los ejemplos\n
    batch_size: numero de ejemplos\n
    nclases: numero de etiquetas posible
    """
    samples = []
    sigma = 1/nclases
    clases = np.random.randint(0, nclases, batch_size)
    clases1hot= onehotify(clases, nclases)
    for clase in clases:        
        mu = clase*sigma+(0.5*sigma)
        s = np.random.normal(loc = mu, scale=sigma, size=dim_latente)
        samples.append(s)
    return np.array(samples), clases1hot

def fake_sampler(imgs:np.ndarray, encoder:Model) -> np.ndarray:
    """
    Devuelve los ejemplos de coordenadas del espacio latente generados por el decodificador.\n
    imgs: array de imagenes a codificar\n
    encoder: encoder
    """
    latent_fake = encoder.predict(imgs["data"])
    return latent_fake

def fake_class_sampler(imgs, encoder, nclases, **kwargs):
    """
    Devuelve uan tupla con los ejemplos de coordenadas del espacio latente generados por el decodificador y su etiqueta en formato onehot.\n
    imgs: array de imagenes a codificar\n
    encoder: encoder
    nclases: numero posible de etiquetas
    """
    latent_fake = encoder.predict(imgs["data"])
    labels = onehotify(imgs["labels"], nclases)
    return latent_fake, labels

# -----------------------------------------------------------
# MUESTREO Y RESULTADOS
# -----------------------------------------------------------

def sample_imgs(dataset:dict, model:Model, epoch:int, nclases:int, sample_size=5, save_imgs=True, show=False, ruta="Resultados/pruebasAAE", nombre="", title=""):
    """
    Muestra/guarda una comparativa entre imagenes originales y regeneradas por el modelo.\n
    dataset: dataset de donde se quieran tomar las imagenes, debe tener forma {"data":[...], "labels":[...]}\n 
    model: autoencoder que regenerará las imagenes\n 
    epoch: epoch en el que se llama a la funcion, sirve para añadirlo al nombre del fichero y seguir el progreso\n 
    nclases: numero de posibles clases, necesario para onehotificar las etiquetas, puede ser 0 si no se usan etiquetas\n 
    sample_size: tamalo de la muestra a regenerar\n 
    save_imgs: si se guarda o no la imagen\n 
    show: si se muestra o no la imagen\n 
    ruta: carpeta raiz para el guardado\n
    nombre: nombre descriptivo\n 
    title: titulo de la imagen
    """
    # Tomamos sample_size imagenes de muestra
    ids = np.random.randint(0,dataset["data"].shape[0], sample_size)
    set={k:v[ids] for k,v in dataset.items()}
    sample = set["data"]
    if nclases > 1:
        model_input=set
        model_input["labels"]=onehotify(model_input["labels"], nclases)
    else:
        model_input=sample
    # Intentamos regenerar las imagenes
    gen_img = model.predict(model_input)
    if len(gen_img.shape)==5:
        gen_img=gen_img[0]
    # Guardamos una grafica con la muestra (arriba) y las imagenes generadas (abajo)
    f, axxs = plt.subplots(2,sample_size)
    if title!="":
        f.suptitle(title+"_"+nombre, fontsize=12)
    for j in range(sample_size):
        axxs[0,j].imshow(sample[j])
        axxs[1,j].imshow(gen_img[j])
    for i in axxs:
        for j in i:
            j.axis("off")
    if save_imgs:
        ruta+="/Output/Regeneracion"
        mkfolders(ruta)
        savefile= ruta+"/"+nombre+"generationCIFAR10_e%d.jpg" % (epoch)
        f.savefig(savefile)
    if show:
        plt.show()
    plt.close()
    
def generate_samples(dim_latente:int, decoder:Model, epoch:int, ruta="Resultados/pruebasAAE", nombre="pAAE", save_imgs=True, show=False):
    """
    Genera un conjunto de imagenes de 5x5 a partir de coordenadas del espacio latente pertenecientes a una distribucion normal.\n
    dim_latente: numero de dimensiones del espacio latente\n 
    decoder: decoder que grnerará las imagenes\n
    epoch: epoch en el que se llama a la funcion, sirve para añadirlo al nombre del fichero y seguir el progreso\n
    ruta: carpeta raiz para el guardado\n
    nombre: nombre descriptivo
    save_imgs: si se guarda o no la imagen\n
    show: si se muestra o no la imagen\n 
    """
    sample_shape = (5,5)
    latent_samples = true_sampler(dim_latente, np.prod(sample_shape))
    samples=decoder.predict(latent_samples)#*0.5 +0.5
    fig, axxs = plt.subplots(sample_shape[0], sample_shape[1])
    s=0
    for i in range(sample_shape[0]):
        for j in range(sample_shape[1]):
            axxs[i,j].imshow(samples[s])
            axxs[i,j].axis("off")
            s+=1
    if save_imgs:
        ruta+="/Output/Progreso"
        mkfolders(ruta)
        fig.savefig(ruta+"\\"+nombre+"progresscifar10_e%d.png" % (epoch))
    if show:
        print("Imagenes generadas desde el espacio latente:")
        plt.show()
    plt.close()

def show_prevResults(ruta:str, nombre="pAAE", epochs=5000):
    """
    Muestra resultados de una ejecucion anterior.\n
    ruta: carpeta raiz donde se guardaron las imagenes\n
    nombre: nombre descriptivo usado al guardar la ejecucion\n
    epochs: epochs de la ejecucion anterior (parte del nombre de algunos ficheros)
    """
    x = Image(filename=ruta+'\\Output/Progreso'+"\\"+nombre+"progresscifar10_e%d.png" % (epochs-1))
    print("Imagenes generadas desde el espacio latente:")
    display(x)

    x = Image(filename=ruta+"\\"+nombre+"progresscifar10_plot.jpg")
    print("Historia del entrenamiento:")
    display(x)

    x = Image(filename=ruta+"/Output/Regeneracion/TRAINSETgenerationCIFAR10_e%d.jpg" % (epochs))
    print("Imagenes regeneradas desde el set de entrenamiento (arriba originales):") 
    display(x) 
     
    x = Image(filename=ruta+"/Output/Regeneracion/TESTSETgenerationCIFAR10_e%d.jpg" % (epochs))         
    print("Imagenes regeneradas desde imagenes nunca vistas por la red (arriba originales):")
    display(x)

    n = np.random.randint(0, 10)
    x = Image(filename=ruta+"/Output/Latente/GenFromLatentTRAINSET label %d.jpg" % (n))         
    print("Imagenes generadas desde el punto medio entre coordenadas latentes de dos imagenes (TrainSet):")
    display(x)
    
    x = Image(filename=ruta+"/Output/Latente/GenFromLatentTESTSET label %d.jpg" % (n))         
    print("Imagenes generadas desde el punto medio entre coordenadas latentes de dos imagenes (TestSet):")
    display(x)
    
def exploraLatente(encoder:Model, decoder:Model, groupPath:str, ruta="Resultados/pruebasAAE", nombre="pAAE"):
    """
    Muestrea la coordenada media entre imagenes con la misma etiqueta.\n
    encoder: encoder\n
    ddecoder: decoder\n
    groupPath: ruta del dataset de imagenes guardado. Es un diccionario donde las keys son las etiquetas y el valor una lista de imagenes\n
    ruta: ruta donde guardar los resultados\n
    nombre: nombre descriptivo
    """
    groups = unpickle(groupPath)
    size = len(groups[0])
    length = len(groups.keys())
    f, axxs = plt.subplots(length,size+1)
    f.set_size_inches(size*2.5, length*1.41)
    f.suptitle("Latent aproximation from " + groupPath, fontsize=12)
    ruta+="/Output/Latente"
    mkfolders(ruta)
    for key in groups.keys():
        imgs = groups[key]
        latent = []
        
        # Obtencion de coordenadas en espacio latente
        #for i in range(size):
        pred = encoder.predict(imgs)
        latent = pred
            #latent.append(pred)
        # Obtenemos la coordenada intermedia
        latent = tf.math.divide(tf.math.add_n(latent),size)
        latent = np.array([latent])
        #Generamos la imagen de esa coordenada
        generated = decoder.predict(latent)

        
        for j in range(size):
            axxs[key][j].imshow(imgs[j])
            axxs[key][j].axis("off")
            axxs[key][j].set_title("Real")
        axxs[key][size].imshow(generated[0])
        axxs[key][size].axis("off")
        axxs[key][size].set_title("Generated")
    
    plt.show()
    plt.close()
        
    f.savefig(ruta+"\\"+nombre+".jpg")
        
def plot_history(history:list, ruta="Resultados/pruebasAAE", nombre="pAAE", title=""):
    """
    Crea una grafica con el historial de losses y accuracy de los modelos durante el entrenamiento.\n
    history: salida de la funcion de entrenamiento\n
    ruta: raiz de la carpeta de los resultados\n
    nombre: nombre descriptivo\n
    title=titulo de la grafica
    """
    disc_loss = history[0]
    disc_acc  = history[1]
    aac_loss1 = history[2]
    aac_loss2 = history[3]

    fig, axxs = plt.subplots(1,3)
    if title != "":
        fig.suptitle(title, fontsize=16)

    fig.set_figwidth(24)
    fig.set_figheight(6)

    axxs[0].set_title("Discriminator")
    axxs[1].set_title("Discriminator")
    axxs[2].set_title("AdversarialAutoencoder")

    axxs[0].set_xlabel("Epoch")
    axxs[1].set_xlabel("Epoch")
    axxs[2].set_xlabel("Epoch")

    axxs[0].plot(disc_loss, label = "Loss")
    axxs[1].plot(disc_acc, label = "Accuracy")

    axxs[2].plot(aac_loss1, label = "Loss_decoder")
    axxs[2].plot(aac_loss2, label = "Loss_discriminator")

    axxs[0].legend()
    axxs[1].legend()
    axxs[2].legend()

    fig.savefig(ruta+"\\"+nombre+"progresscifar10_plot.jpg")
    plt.show()
    plt.close()

# -----------------------------------------------------------
# ENTRENAMIENTO DE REDES
# -----------------------------------------------------------

def fit_AAE_twoPhased(dim_latente:int, aae:tuple, dataset:dict, epochs=12, batch_size=100, sample_interval=100, ruta="Resultados/pruebasAAE", nombre="pAAE", verbose=True,
            truth=true_sampler, truth_kwargs={}, falsehood=fake_sampler):
    """
    Entrenamiento en dos fases, devuelve historial de entrenamiento. 1-Discriminador 2-Autoencoder. 
    En la fase dos se entrenan al mismo tiempo la regeneracion de las imagenes y el ajuste del espacio latente a la distribucion en la que se haya entrenado el discriminador.\n
    dim_latente: dimensiones del espacio latente \n
    aae: tupla con los modelos\n
    dataset: conjunto de datos de entrenamiento\n
    epochs: numero de epochs\n
    batch_size: tamaño del lote de imagenes con las que entrenar al mismo tiempo\n
    sample_interval: intervalo de salida por consola de datos\n
    ruta: raiz para el guardado de resultados\n
    nombre: nombre descriptivo\n
    verbose: si imprime salida por consola o no\n
    truth: funcion de la distribucion que debe seguir el discriminador\n
    truth_kwargs: argumentos para la funcion de la distribucion\n
    falsehood: funcion de toma de muestras con el encoder
    """
    elements = len(dataset["data"])
    totalSteps = ceil(elements/batch_size)

    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(elements, seed=2022)
    dataset = dataset.batch(batch_size)

    history = np.empty([0,4])

    encoder, decoder, discriminator, a_autoencoder=aae
    
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    truth_params = {"dim_latente":dim_latente, "batch_size":batch_size}
    truth_params.update(truth_kwargs)

    print("Entrenando: "+ruta)

    for epoch in range(epochs):
        start = timer()
        for step, imgs in enumerate(dataset):
            falsehood_params = {"imgs":imgs, "encoder":encoder}
            falsehood_params.update(truth_kwargs)
            #Espacios latentes "reales" y "falsos" para el discriminador
            latent_fake = falsehood(**falsehood_params)
            latent_true = truth(**truth_params)

            #entrenamos el discriminador
            dis_loss_real = discriminator.train_on_batch(latent_true, valid)
            dis_loss_fake = discriminator.train_on_batch(latent_fake, fake)
            dis_avg_loss = 0.5*np.add(dis_loss_fake, dis_loss_real)

            # entrenamos al autoencoder
            if "nclases" in truth_kwargs:
                imgs["labels"]=onehotify(imgs["labels"], truth_kwargs["nclases"])
            aae_loss = a_autoencoder.train_on_batch(imgs,[imgs["data"], valid]) # El resultado debe ser la imagen sin las etiquetas
            
            # Guardamos el progreso
            history = np.append(history, [np.append(dis_avg_loss[:2], aae_loss[:2])], axis=0)
            
            # monitorizamos el progreso
            if step*((step+1) % sample_interval)==0 and verbose:
                progressPercent=step/totalSteps
                bar=ceil(progressPercent*10)
                elapsed = timer() - start
                print("E%d <" % (epoch)+chr(9608)*bar+" "*(10-bar)+"> %d%% DISC: [loss: %f, acc: %.2f%%] AAE: [mse: %f, b_ce: %f] %.2fs\t\t" % (ceil(100*progressPercent), dis_avg_loss[0], 100*dis_avg_loss[1], aae_loss[0], aae_loss[1], elapsed), end="\r")
        if verbose:
            print("")
        # Hacemos una muestra visual
        generate_samples(dim_latente, decoder, epoch, ruta=ruta, nombre=nombre, show=((epoch+1)==epochs))
    return history.transpose(1,0)

def fit_AAE_threePhased(dim_latente:int, aae:tuple, dataset:dict, epochs=12, batch_size=100, sample_interval=100, ruta="Resultados/pruebasAAE", nombre="pAAE", verbose=True,
            truth=true_sampler, truth_kwargs={}, falsehood=fake_sampler):
    """
    Entrenamiento en dos fases, devuelve historial de entrenamiento. 1-Discriminador 2-Autoencoder. 
    En la fase dos se entrenan al mismo tiempo la regeneracion de las imagenes y el ajuste del espacio latente a la distribucion en la que se haya entrenado el discriminador.\n
    dim_latente: dimensiones del espacio latente \n
    aae: tupla con los modelos\n
    dataset: conjunto de datos de entrenamiento\n
    epochs: numero de epochs\n
    batch_size: tamaño del lote de imagenes con las que entrenar al mismo tiempo\n
    sample_interval: intervalo de salida por consola de datos\n
    ruta: raiz para el guardado de resultados\n
    nombre: nombre descriptivo\n
    verbose: si imprime salida por consola o no\n
    truth: funcion de la distribucion que debe seguir el discriminador\n
    truth_kwargs: argumentos para la funcion de la distribucion\n
    falsehood: funcion de toma de muestras con el encoder
    """
    elements = len(dataset["data"])
    totalSteps = ceil(elements/batch_size)

    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(elements, seed=2022)
    dataset = dataset.batch(batch_size)

    history = np.empty([0,4])

    encoder, decoder, discriminator, autoencoder, encoscriminador=aae
    
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    print("Entrenando: "+ruta)
    
    for epoch in range(epochs):
        start = timer()
        for step, imgs in enumerate(dataset):
            #Espacios latentes "reales" y "falsos" para el discriminador
            latent_fake = falsehood(imgs=imgs, encoder=encoder, **truth_kwargs)
            latent_true = truth(dim_latente=dim_latente, batch_size=batch_size, **truth_kwargs)

            #entrenamos el discriminador
            dis_loss_real = discriminator.train_on_batch(latent_true, valid)
            dis_loss_fake = discriminator.train_on_batch(latent_fake, fake)
            dis_avg_loss = 0.5*np.add(dis_loss_fake, dis_loss_real)

            # entrenamos al regenerador
            if "nclases" in truth_kwargs:
                imgs["labels"]=onehotify(imgs["labels"], truth_kwargs["nclases"])
            ae_loss = autoencoder.train_on_batch(imgs, imgs["data"]) 
            
            # Entrenamos el generador, la idea es que el encoder genere "imagenes validas"
            ed_loss=encoscriminador.train_on_batch(imgs, valid)

            # Guardamos el progreso
            history = np.append(history, [np.append(dis_avg_loss[:2], [ae_loss, ed_loss[1]])], axis=0)
            
            # monitorizamos el progreso
            if step*((step+1) % sample_interval)==0 and verbose:
                progressPercent=step/totalSteps
                bar=ceil(progressPercent*10)
                elapsed = timer() - start
                print("E%d <" % (epoch)+chr(9608)*bar+" "*(10-bar)+"> %d%% DISC: [loss: %f, acc: %.2f%%] AE: [loss: %f] EN-DI [acc: %.2f%%] %.2fs\t\t" % (ceil(100*progressPercent), dis_avg_loss[0], 100*dis_avg_loss[1], ae_loss, 100*ed_loss[1], elapsed), end="\r")
        if verbose:
            print("")
        # Hacemos una muestra visual
        generate_samples(dim_latente, decoder, epoch, ruta=ruta, nombre=nombre, show=((epoch+1)==epochs))
    
    return history.transpose(1,0)