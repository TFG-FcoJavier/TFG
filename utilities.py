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
