import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras import Input, Model, layers

def load_images(input_dir: str):
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith('.png')]
    imgs = []
    for fname in files:
        img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(arr)
    if not imgs:
        return np.empty((0,)), []
    return np.stack(imgs, axis=0), files

def train_autoencoder(input_dir: str,
                      latent_dim: int = 64,
                      epochs: int = 50,
                      test_split: float = 0.2):
    """
    Trainiert den Autoencoder und verwendet einen Teil der Bilder zum Testen.
    test_split: Anteil der Bilder für das Testset (z.B. 0.2 = 20%).
    """
    data, files = load_images(input_dir)
    if data.size == 0:
        raise RuntimeError("Keine Bilder zum Trainieren gefunden.")

    # Aufteilen in Training und Test
    num = data.shape[0]
    idx = np.random.permutation(num)
    split = int(num * (1 - test_split))
    train_idx, test_idx = idx[:split], idx[split:]
    x_train, x_test = data[train_idx], data[test_idx]

    h, w, c = data.shape[1:]
    ae, encoder, decoder = build_autoencoder((h, w, c), latent_dim)
    ae.compile(optimizer='adam', loss='mse')

    # Training
    history = ae.fit(
        x_train, x_train,
        epochs=epochs,
        validation_data=(x_test, x_test),
        verbose=1
    )
    print(f"Trained on {x_train.shape[0]} images, validated on {x_test.shape[0]} images.")
    return ae, encoder, decoder, history, x_test


def compare_original_and_compressed(orig_dir: str,
                                    comp_dir: str,
                                    max_examples: int = 5):
    orig_files = [f for f in os.listdir(orig_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pairs = []
    for fn in orig_files:
        base = os.path.splitext(fn)[0]
        comp_path = os.path.join(comp_dir, base + '.png')
        orig_path = os.path.join(orig_dir, fn)
        if os.path.isfile(comp_path):
            pairs.append((orig_path, comp_path))
    pairs = pairs[:max_examples]

    if not pairs:
        print("Keine Bildpaare zum Vergleichen gefunden.")
        return

    n = len(pairs)
    plt.figure(figsize=(6, 3 * n))
    for i, (orig_path, comp_path) in enumerate(pairs):
        orig = np.array(Image.open(orig_path).convert('RGB'), dtype=np.float32)
        comp = np.array(Image.open(comp_path).convert('RGB'), dtype=np.float32)
        mse = np.mean((orig - comp) ** 2)
        psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse > 0 else np.inf

        ax1 = plt.subplot(n, 2, 2*i + 1)
        ax1.imshow(orig.astype(np.uint8))
        ax1.set_title(f"Original\n{os.path.basename(orig_path)}")
        ax1.axis('off')

        ax2 = plt.subplot(n, 2, 2*i + 2)
        ax2.imshow(comp.astype(np.uint8))
        ax2.set_title(f"Compressed\nPSNR = {psnr:.1f} dB")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

def build_autoencoder(input_shape: tuple, latent_dim: int):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPool2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2, padding='same')(x)
    x = layers.Flatten()(x)
    code = layers.Dense(latent_dim, activation='relu')(x)
    encoder = Model(inp, code, name='encoder')

    dec_inp = Input(shape=(latent_dim,))
    x = layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64, activation='relu')(dec_inp)
    x = layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64))(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    dec_out = layers.Conv2DTranspose(input_shape[2], 3, activation='sigmoid', padding='same')(x)
    decoder = Model(dec_inp, dec_out, name='decoder')

    autoencoder = Model(inp, decoder(encoder(inp)), name='autoencoder')
    return autoencoder, encoder, decoder


def simple_compress_folder(input_dir: str,
                           output_dir: str,
                           target_size_kb: int = 50,
                           max_images: int = None,
                           min_quality: int = 5,
                           max_quality: int = 95):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images is not None:
        files = files[:max_images]

    for fname in files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, os.path.splitext(fname)[0] + '.jpeg')
        img = Image.open(src).convert('RGB')

        lower, upper = min_quality, max_quality
        final_quality = lower

        while lower <= upper:
            quality = (lower + upper) // 2
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            size_kb = len(buf.getvalue()) / 1024

            if abs(size_kb - target_size_kb) <= 1:
                final_quality = quality
                break

            if size_kb > target_size_kb:
                upper = quality - 1
            else:
                lower = quality + 1
                final_quality = quality

        buf = BytesIO()
        img.save(buf, format='JPEG', quality=final_quality)
        with open(dst, 'wb') as f:
            f.write(buf.getvalue())

        final_size_kb = os.path.getsize(dst) / 1024
        print(f"Compressed {fname} -> {dst} (target={target_size_kb:.1f} KB, actual={final_size_kb:.1f} KB, quality={final_quality})")


if __name__ == '__main__':
    src_folder = 'input/genuine'
    compressed_folder = 'output/compressed'
    target_size_kb = 10
    max_images = 20
    latent_dim = 64
    epochs = 10
    test_split = 0.2

    simple_compress_folder(
        src_folder,
        compressed_folder,
        target_size_kb=target_size_kb,
        max_images=max_images
    )


   # 2) Training mit Test-Split
    ae, encoder, decoder, history, x_test = train_autoencoder(
        compressed_folder,
        latent_dim=latent_dim,
        epochs=epochs,
        test_split=test_split
    )