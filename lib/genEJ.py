"""
GENERACION DE EJEMPLOS
"""

import numpy as np
import numpy.random as random
from keras import Model
from math import floor

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

def true_multivariate_sampler(dim_latente, batch_size, nclases, center_margin=(0,0), cov_chance=0.8, **kwargs):
    samples = []
    clases = random.randint(0, nclases, batch_size)
    clases1hot= onehotify(clases, nclases)

    mus = []
    covMatrixes = []
    for i in range(nclases):
        # Reproducible random state para cada etiqueta
        generator = random.default_rng(i)
        # Centro de la distribucion
        mu = generator.random(dim_latente)
        mu = [(20 * i) -10 for i in mu]
        mus.append(mu)
        # Matriz de covarianza
        covMatrix = np.zeros((dim_latente, dim_latente))
        diag = generator.random(dim_latente)
        diag = [1 if i <cov_chance else .2 for i in diag]
        np.fill_diagonal(covMatrix, diag)
        covMatrixes.append(covMatrix)

    for label in clases:
        # Genera la muestra para el ejemplo con esa etiqueta
        samples.append(random.multivariate_normal(mus[label], covMatrixes[label]))

    return np.array(samples), clases1hot

try:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    def true_multivariate_sampler_tf(dim_latente, batch_size, nclases, cov_chance=0.8, **kwargs):

        sPerC = floor(batch_size/nclases)
        samples = None
        clases = np.array([])

        for i in range(nclases):
            # Reproducible random state para cada etiqueta
            generator = random.default_rng(i)
            # Centro de la distribucion
            mu = generator.random(dim_latente)
            mu = [(20 * i) -10 for i in mu]

            # Matriz de covarianza
            #covMatrix = np.zeros((dim_latente, dim_latente))
            diag = generator.random(dim_latente)
            diag = [1 if i <cov_chance else .2 for i in diag]
            #np.fill_diagonal(covMatrix, diag)

            # Representacion de la normal
            mvn = tfd.MultivariateNormalDiag(
                loc=mu,
                scale_diag=diag)

            # Muestras
            #s=random.multivariate_normal(mu, covMatrix, size=sPerC)
            s= mvn.sample(sample_shape=(sPerC))
            if samples is None:
                samples=s
            else:
                samples=np.append(samples, s, axis=0)
            clases=np.append(clases, np.ones(sPerC)*i)
            
        random.default_rng(2022).shuffle(samples)
        random.default_rng(2022).shuffle(clases)

        return samples, onehotify(clases.astype(int), nclases)
except Exception as e:
    print(e)
    print("Instala la libreria de tensorflow_probability para acelerar la generacion de muestras de la multivariada")
    def true_multivariate_sampler_tf(dim_latente, batch_size, nclases, cov_chance=0.8, **kwargs):

        sPerC = floor(batch_size/nclases)
        samples = None
        clases = np.array([])

        for i in range(nclases):
            # Reproducible random state para cada etiqueta
            generator = random.default_rng(i)
            # Centro de la distribucion
            mu = generator.random(dim_latente)
            mu = [(20 * i) -10 for i in mu]

            # Matriz de covarianza
            covMatrix = np.zeros((dim_latente, dim_latente))
            diag = generator.random(dim_latente)
            diag = [1 if i <cov_chance else .2 for i in diag]
            np.fill_diagonal(covMatrix, diag)

            # Muestras
            s=random.multivariate_normal(mu, covMatrix, size=sPerC)
            if samples is None:
                samples=s
            else:
                samples=np.append(samples, s, axis=0)
            clases=np.append(clases, np.ones(sPerC)*i)
            
        random.default_rng(2022).shuffle(samples)
        random.default_rng(2022).shuffle(clases)
        return samples, onehotify(clases.astype(int), nclases)

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