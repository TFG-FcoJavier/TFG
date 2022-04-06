"""
MUESTREO Y RESULTADOS
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras import Model
from IPython. display import Image, display

from utilities import mkfolders, unpickle
from genEJ import onehotify, true_sampler


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
    if type(gen_img) is list:
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
        
def plot_history(history:dict, ruta="Resultados/pruebasAAE", nombre="pAAE", title=""):
    """
    Crea una grafica con el historial de losses y accuracy de los modelos durante el entrenamiento.\n
    history: salida de la funcion de entrenamiento\n
    ruta: raiz de la carpeta de los resultados\n
    nombre: nombre descriptivo\n
    title=titulo de la grafica
    """
    metricas = history.keys()
    len_m = len(metricas)

    fig, axxs = plt.subplots(1, len_m)
    if title != "":
        fig.suptitle(title, fontsize=16)

    fig.set_figwidth(8*len_m)
    fig.set_figheight(6)

    for i, metrica in enumerate(metricas):
        axxs[i].set_title(metrica)
        axxs[i].set_xlabel("Epoch")
        for modelo, datos in history[metrica].items():
            axxs[i].plot(datos, label = modelo)
        axxs[i].legend()

    fig.savefig(ruta+"\\"+nombre+"progresscifar10_plot.jpg")
    plt.show()
    plt.close()