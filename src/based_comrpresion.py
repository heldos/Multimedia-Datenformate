import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from io import BytesIO
import os

def build_autoencoder(input_shape: tuple, latent_dim: int) -> tuple[Model, Model, Model]:
    inp = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPool2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2, padding='same')(x)
    x = layers.Flatten()(x)
    code = layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    x = layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64, activation='relu')(code)
    x = layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64))(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    out = layers.Conv2DTranspose(input_shape[2], 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inp, out, name='autoencoder')
    encoder = Model(inp, code, name='encoder')

    # Decoder-Model aus den letzten Layers des Autoencoders
    code_in = layers.Input(shape=(latent_dim,))
    dec_x = autoencoder.layers[-4](code_in)
    for layer in autoencoder.layers[-3:]:
        dec_x = layer(dec_x)
    decoder = Model(code_in, dec_x, name='decoder')

    return autoencoder, encoder, decoder

def compress_image(input_path: str,
                   output_path: str,
                   latent_dim: int = 64,
                   epochs: int = 50,
                   target_size: int = None,
                   quality_step: int = 5,
                   min_quality: int = 20) -> str:
    """
    Komprimiere Bild mit Autoencoder und optionaler Größenkontrolle.
    Wenn target_size gesetzt ist, wird die JPEG-Qualität iterativ reduziert.
    """
    # 1) Bild laden und vorverarbeiten
    img = Image.open(input_path).convert('RGB')
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0
    x = arr[np.newaxis, ...]

    # 2) Autoencoder erstellen und trainieren
    autoencoder, encoder, decoder = build_autoencoder(x.shape[1:], latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, epochs=epochs, verbose=0)

    # 3) Kompression & Rekonstruktion
    code = encoder.predict(x)
    recon = decoder.predict(code)[0]

    # 4) In uint8 zurück und PIL-Image
    recon_img = (recon * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(recon_img)

    # 5) Wenn keine Größenvorgabe → direkt speichern
    if target_size is None:
        pil_img.save(output_path, format='JPEG')
        return output_path

    # 6) Iterative Qualitätsanpassung
    buffer = BytesIO()
    quality = 95
    pil_img.save(buffer, format='JPEG', quality=quality)
    size = buffer.tell()

    while size > target_size and quality >= min_quality:
        buffer.seek(0)
        buffer.truncate(0)
        quality -= quality_step
        pil_img.save(buffer, format='JPEG', quality=quality)
        size = buffer.tell()

    # 7) Endgültig auf Platte schreiben
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())

    print(f"Final quality={quality}, file size={size} bytes")
    return output_path

# Beispielaufruf:
#if __name__ == "__main__":
 #   out = compress_image(
  #      input_path="input.png",
  #      output_path="output/compressed.jpg",
  #      latent_dim=64,
  #      epochs=50,
   #     target_size=50_000,    # Zielgröße in Bytes (z.B. 50 KB)
  #      quality_step=5,        # Qualität wird in 5er-Schritten reduziert
  #      min_quality=20         # Unterschreitung der Minimalqualität abwarten
  #  )
  #  print("Gespeichert unter:", out)
