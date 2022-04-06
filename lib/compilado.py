"""
COMPILADO DE MODELOS
"""
import keras
from keras.layers import Input
from keras import Model
from keras import losses

from densa import build_dense_decoder, build_dense_encoder
from discriminadores import build_discriminator

def assemble_AAE_twoPhased(dim_latente:int, img_shape:tuple, enc_model:Model=build_dense_encoder, enc_kwargs:dict={}, dec_model:Model=build_dense_decoder, dec_kwargs:dict={}, disc_model:Model = build_discriminator, disc_kwargs:dict={}, ae_loss=losses.mean_squared_error, disc_loss= losses.binary_crossentropy, optimizer = keras.optimizers.Adam(0.0002, 0.5), loss_weights=[1, 1]) -> tuple:
    """
    Metodo para la construccion. Devuelve una tupla con los modelos del encoder, decoder, discriminador y autoencoder adversario en ese orden.\n
    dim_latente: dimensiones del espacio latente \n
    img_shape: tamaño de la imagen \n
    enc_model: funcion de construccion del encoder \n
    dec_model: funcion de construccion del decoder\n
    disc_model: funcion de construccion del discriminador \n
    enc_kwargs: argumentos admitidos por el encoder \n
    dec_kwargs: argumentos admitidos por el decoder\n
    disc_kwargs: argumentos admitidos por el discriminador
    """
    # Discriminador
    discriminator = disc_model(dim_latente=dim_latente, **disc_kwargs)
    discriminator.compile(loss=disc_loss, metrics="accuracy")

    # Encoder y decoder
    encoder = enc_model(dim_latente=dim_latente,img_shape=img_shape, **enc_kwargs)
    decoder = dec_model(dim_latente=dim_latente,img_shape=img_shape, **dec_kwargs)

    # Autoencoder
    # el encoder toma un imagen y la codifica y el decoder toma la codificacion e intenta regenerar la imagen
    img = Input(shape=img_shape, name="data")
    encoded = encoder(img)
    reconstructed = decoder(encoded)
    
    if "clases" in disc_kwargs.keys():
        clase = Input(shape=disc_kwargs["clases"], name = "labels")
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
    a_autoencoder.compile(loss=[ae_loss, disc_loss], optimizer=optimizer, loss_weights=loss_weights)
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
    encosriminator.compile(optimizer=cp["optimizer"], loss=cp["disc_loss"], metrics=['accuracy'])

    return (encoder, decoder, discriminator, autoencoder, encosriminator)

