import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(input_shape: tuple, latent_dim: int) -> tuple[Model, Model, Model]:

    # --- Encoder-Aufbau ---
    inp = layers.Input(shape=input_shape)  # Eingabe-Layer für Bilddaten

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    # Max-Pooling halbiert die räumliche Auflösung
    x = layers.MaxPool2D(2, padding='same')(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2, padding='same')(x)  # zweite Pooling-Stufe
    x = layers.Flatten()(x)  # Merkmalskarten zu Vektor zusammenfassen
    code = layers.Dense(latent_dim, activation='relu')(x)  # Latent-Code-Dimension

    x = layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64, activation='relu')(code)

    x = layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64))(x)
    # Upsampling verdoppelt die räumliche Auflösung
    x = layers.UpSampling2D(2)(x)
    # Transponierte Faltung, um Details zu lernen
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)

    out = layers.Conv2DTranspose(input_shape[2], 3, activation='sigmoid', padding='same')(x)


    autoencoder = Model(inp, out, name='autoencoder')
    encoder = Model(inp, code, name='encoder')
    code_in = layers.Input(shape=(latent_dim,))
    dec_x = autoencoder.layers[-4](code_in)

    for layer in autoencoder.layers[-3:]:
        dec_x = layer(dec_x)
    decoder = Model(code_in, dec_x, name='decoder')

    return autoencoder, encoder, decoder


# Diese Funktion führt die Kompression und Rekonstruktion auf einem Bild durch.
def compress_image(input_path: str, output_path: str, latent_dim: int = 64, epochs: int = 50) -> str:

    img = Image.open(input_path).convert('RGB')
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0
    x = arr[np.newaxis, ...]

    # Autoencoder erzeugen und kompilieren
    autoencoder, encoder, decoder = build_autoencoder(x.shape[1:], latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, epochs=epochs, verbose=0)

    # Kompression: Encoder erstellt den latenten Code
    code = encoder.predict(x)
    # Rekonstruktion: Decoder wandelden Code zurück in ein Bild
    recon = decoder.predict(code)[0]

    # Rekonstruiertes Bild zurück in 0-255 Werte und uint8
    recon_img = (recon * 255).clip(0, 255).astype(np.uint8)
    compressed = Image.fromarray(recon_img)

    # Komprimiertes Bild speichern
    compressed.save(output_path)
    return output_path
