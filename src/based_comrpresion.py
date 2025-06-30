import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from io import BytesIO
import os

def build_autoencoder(input_shape: tuple, latent_dim: int):
    # Encoder
    encoder_input = Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name="enc_conv1")(encoder_input)
    x = layers.MaxPool2D(2, padding='same', name="enc_pool1")(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name="enc_conv2")(x)
    x = layers.MaxPool2D(2, padding='same', name="enc_pool2")(x)
    x = layers.Flatten(name="enc_flatten")(x)
    code = layers.Dense(latent_dim, activation='relu', name="bottleneck")(x)

    encoder = Model(encoder_input, code, name='encoder')

    # Decoder-Layer instanziieren (einmalig)
    d_dense   = layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64,
                              activation='relu', name="dec_dense")
    d_reshape = layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64), name="dec_reshape")
    d_upsamp1 = layers.UpSampling2D(2, name="dec_upsample1")
    d_deconv1 = layers.Conv2DTranspose(32, 3, activation='relu', padding='same', name="dec_deconv1")
    d_upsamp2 = layers.UpSampling2D(2, name="dec_upsample2")
    d_deconv2 = layers.Conv2DTranspose(input_shape[2], 3, activation='sigmoid', padding='same', name="dec_deconv2")

    # Autoencoder-Graph zusammenführen
    x_dec = d_dense(code)
    x_dec = d_reshape(x_dec)
    x_dec = d_upsamp1(x_dec)
    x_dec = d_deconv1(x_dec)
    x_dec = d_upsamp2(x_dec)
    decoded = d_deconv2(x_dec)
    autoencoder = Model(encoder_input, decoded, name='autoencoder')

    # Separates Decoder-Model mit den gleichen Layer-Instanzen
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    y = d_dense(decoder_input)
    y = d_reshape(y)
    y = d_upsamp1(y)
    y = d_deconv1(y)
    y = d_upsamp2(y)
    decoder_output = d_deconv2(y)
    decoder = Model(decoder_input, decoder_output, name='decoder')

    return autoencoder, encoder, decoder

def compress_image(input_path: str,
                   output_path: str,
                   latent_dim: int = 64,
                   epochs: int = 50,
                   target_size: int = None,
                   quality_step: int = 5,
                   min_quality: int = 20) -> str:
    """
    Komprimiere Bild mit Autoencoder und optionaler JPEG-Größenkontrolle.
    Die Original-Dimensionen bleiben erhalten (vorausgesetzt H, W % 4 == 0).
    """
    # 1) Bild laden (keine Verzerrung!)
    img = Image.open(input_path).convert('RGB')
    width, height = img.size

    # 2) Normieren und Batch-Dim hinzufügen
    arr = np.array(img) / 255.0
    x = arr[np.newaxis, ...]  # Form (1, H, W, 3)

    # 3) Autoencoder bauen & trainieren
    autoencoder, encoder, decoder = build_autoencoder((height, width, 3), latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, epochs=epochs, verbose=0)

    # 4) Kodieren & Rekonstruieren
    code = encoder.predict(x)
    recon = decoder.predict(code)[0]

    # 5) Zurück zu uint8 + PIL
    recon_img = (recon * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(recon_img)

    # 6) Wenn keine Zielgröße, direkt speichern
    if target_size is None:
        pil.save(output_path, 'JPEG')
        return output_path

    # 7) Iterative Qualitäts-Reduktion
    buf = BytesIO()
    quality = 95
    pil.save(buf, 'JPEG', quality=quality)
    size = buf.tell()

    while size > target_size and quality >= min_quality:
        buf.seek(0); buf.truncate(0)
        quality -= quality_step
        pil.save(buf, 'JPEG', quality=quality)
        size = buf.tell()

    # 8) Endgültig schreiben
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(buf.getvalue())

    print(f"Final quality={quality}, size={size} bytes")
    return output_path

if __name__ == "__main__":
    out = compress_image(
        input_path="test.png",
        output_path="output/compressed.jpg",
        latent_dim=64,
        epochs=50,
        target_size=50_000,
        quality_step=5,
        min_quality=20
    )
    print("Gespeichert unter:", out)
