"""
ENTRENAMIENTO DE REDES
"""

import tensorflow as tf
import numpy as np

from math import ceil
from timeit import default_timer as timer

from lib.genEJ import *
from lib.muestreo import generate_samples

def fit_AAE_twoPhased(dim_latente:int, aae:tuple, dataset:dict, epochs=12, batch_size=100, sample_interval=100, ruta="Resultados/pruebasAAE", nombre="pAAE", verbose=True,
            truth=true_sampler, truth_kwargs={}, falsehood=fake_sampler) -> dict:
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

    history = {
        "loss":{
            "discriminador": np.array([]),
            "AAE_Discrim": np.array([]),
            "AAE_Decoder": np.array([])
            }, 
        "accuracy":{"discriminador": np.array([])}
        }

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
            history["loss"]["discriminador"]=np.append(history["loss"]["discriminador"],dis_avg_loss[0])
            history["loss"]["AAE_Discrim"]=np.append(history["loss"]["AAE_Discrim"],aae_loss[1])
            history["loss"]["AAE_Decoder"]=np.append(history["loss"]["AAE_Decoder"],aae_loss[0])
            history["accuracy"]["discriminador"]=np.append(history["accuracy"]["discriminador"],dis_avg_loss[1])
            
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
    return history

def fit_AAE_threePhased(dim_latente:int, aae:tuple, dataset:dict, epochs=12, batch_size=100, sample_interval=100, ruta="Resultados/pruebasAAE", nombre="pAAE", verbose=True,
            truth=true_sampler, truth_kwargs={}, falsehood=fake_sampler) -> dict:
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

    history = {
        "loss":{
            "discriminador": np.array([]),
            "autoencoder": np.array([]),
            "encoder+discr": np.array([])
            }, 
        "accuracy":{
            "discriminador": np.array([]),
            "encoder+discr": np.array([])
            }
        }

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
            history["loss"]["discriminador"]=np.append(history["loss"]["discriminador"],dis_avg_loss[0])
            history["loss"]["autoencoder"]=np.append(history["loss"]["autoencoder"],ae_loss)
            history["loss"]["encoder+discr"]=np.append(history["loss"]["encoder+discr"],ed_loss[0])
            history["accuracy"]["discriminador"]=np.append(history["accuracy"]["discriminador"],dis_avg_loss[1])
            history["accuracy"]["encoder+discr"]=np.append(history["accuracy"]["encoder+discr"],ed_loss[1])
            
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
    
    return history