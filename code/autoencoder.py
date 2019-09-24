import tensorflow as tf
import numpy as np
import util
import matplotlib.pyplot as plt
import io
from PIL import Image
import os


def plot_results(models, data, batch_size=128, model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(filename)
    plt.show()


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


def sampling2(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + z_log_var * epsilon


from callbacks.AutoEncoderSummary import AutoEncoderSummary

x_train, y_train, x_test, y_test = util.getKaggleMNIST()

original_dim = x_train.shape[1]

# network parameters
input_shape = (original_dim,)
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)

var_activation = tf.keras.activations.softplus
z_log_var = tf.keras.layers.Dense(
    latent_dim, name="z_log_var", activation=var_activation
)(x)

# use reparameterization trick to push the sampling out as input
z = tf.keras.layers.Lambda(sampling2, output_shape=(latent_dim,), name="z")(
    [z_mean, z_log_var]
)

# instantiate encoder model
encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
# tf.keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name="z_sampling")
x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = tf.keras.layers.Dense(original_dim, activation="sigmoid")(x)

# instantiate decoder model
decoder = tf.keras.models.Model(latent_inputs, outputs, name="decoder")
decoder.summary()
# tf.keras.utils.plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.models.Model(inputs, outputs, name="vae_mlp")

models = (encoder, decoder)
data = (x_test, y_test)

# reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim

kl_loss = -tf.keras.backend.log(z_log_var) + 0.5 * (
    tf.keras.backend.pow(z_log_var, 2) + tf.keras.backend.pow(z_mean, 2)
)
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss += -0.5

vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer="adam")
vae.summary()


data_to_plot = []
index = np.random.choice(range(x_train.shape[0]), 10)
data_train = x_train[index]
index = np.random.choice(range(x_test.shape[0]), 10)
data_test = x_test[index]
data_to_plot.append((data_train, "train"))
data_to_plot.append((data_test, "valid"))

summaries_dir = "summaries/vae_kaggle_mnist"
tf.gfile.DeleteRecursively(summaries_dir) if tf.gfile.Exists(summaries_dir) else None
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=summaries_dir)

vae.fit(
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None),
    callbacks=[
        tensorboard_callback,
        # AutoEncoderSummary(tensorboard_callback, data_to_plot, update_freq=1),
    ],
)


plot_results(models, data, batch_size=batch_size, model_name="vae_mlp")
