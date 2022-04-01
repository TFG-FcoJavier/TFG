import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def mkfolders(ruta, verbose=False):
    try:
        os.makedirs(ruta)
    except OSError:
        if verbose: print("Carpeta %s ya existe" % (ruta))

# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin')
    return dict

def saveDataset(dataset, name, input_path,dataset_name):
    savefile = input_path+dataset_name+"-"+name
    if os.path.isfile(savefile):
        print("Este set ya esta preparado")
    else:
        pickle.dump(dataset, open(savefile, "wb"))

def tryDataset(dataset):
    print(dataset["data"].shape)
    #plt.figure()
    f, arrx = plt.subplots(1,2)
    i = np.random.randint(0,dataset["data"].shape[0])
    arrx[0].imshow(dataset["data"][i])
    arrx[0].set_title(dataset["labels"][i])
    i = np.random.randint(0,dataset["data"].shape[0])
    arrx[1].imshow(dataset["data"][i])
    arrx[1].set_title(dataset["labels"][i])
    for a in arrx:
        a.axis("off")
    plt.show()
    plt.close()

def compute_mixture(dataset, encoder, points = None, override=False, ruta="Data/", nombre="mixtureData", dataset_name=""):
    exists=os.path.isfile(ruta+dataset_name+"-"+nombre)
    
    if exists and not override:
        return load_mixture(ruta=ruta, nombre=nombre, dataset_name=dataset_name)
    
    if points is None:
        points = encoder.predict(dataset["data"])
        
    mixture = {}
    for i, label in enumerate(dataset["labels"]):
        if label in mixture.keys():
            mixture[label].append(points[i])
        else:
            mixture[label]= [points[i]]
    for index in mixture.keys():
        cluster = mixture[index]
        mixture[index] = {"mu":[], "sigma":[]}
        mixture[index]["mu"]=np.median(cluster, axis=0)
        mixture[index]["sigma"]=np.std(cluster, axis=0)
        
    if not exists or override:    
        print("Guardando datos de la mixtura para "+dataset_name+"-"+nombre)
        mkfolders(ruta=ruta)
        pickle.dump(mixture, open(ruta+dataset_name+"-"+nombre, "wb"))
    return mixture
    
def load_mixture(ruta="Data/", nombre="mixtureData", dataset_name="", expansion=1, **kwargs):
    mixture = unpickle(ruta+dataset_name+"-"+nombre)
    for v in mixture.values():
        v["mu"] = np.multiply(v["mu"], expansion)
    return mixture