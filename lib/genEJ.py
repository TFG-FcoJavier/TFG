"""
GENERACION DE EJEMPLOS
"""

import numpy as np
from keras import Model

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

def genFromMixture(dim_latente, batch_size, nclases, mixture):
    """
    Devuelve un array de numpy que contiene ejemplos de coordenadas del espacio latente, cada una pertenecientes a una distribucion normal distinta dependiendo de su etiqueta.
    Estas distribuciones se basan em los datos de una mixtura generados con antelacion.\n
    dim_latente: tamaño de los ejemplos\n
    batch_size: numero de ejemplos\n
    nclases: numero de etiquetas posible\n
    mixture: diccionario mixtura
    """
    samples = []
    clases = np.random.randint(0, nclases, batch_size)
    clases1hot= onehotify(clases, nclases)
    for clase in clases: 
        s = np.random.normal(loc = mixture[clase]["mu"], scale=mixture[clase]["sigma"], size=dim_latente)
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