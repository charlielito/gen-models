import tensorflow as tf


class VEncoder(tf.keras.Model):
    def __init__(self, hidden_layers_sizes, activation=tf.nn.relu):
        super().__init__()
        self.activation = activation
        self.net = tf.keras.models.Sequential()

        for layer_size in hidden_layers_sizes[:-1]:
            layer = tf.keras.layers.Dense(layer_size, activation=self.activation)
            self.net.add(layer)

        z_space_size = hidden_layers_sizes[-1]
        self.mean_layer = tf.keras.layers.Dense(z_space_size)
        self.std_layer = tf.keras.layers.Dense(z_space_size, activation=None)

    def call(self, inputs):
        x = self.net(inputs)
        means = self.mean_layer(x)
        stds = self.std_layer(x)

        # reparameterization trick
        z_sample = sample(means, stds)
        return means, stds, z_sample


def sample(means, stds):
    batch = tf.keras.backend.shape(means)[0]
    dim = tf.keras.backend.int_shape(means)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return means + tf.keras.backend.exp(0.5 * stds) * epsilon


class Decoder(tf.keras.Model):
    def __init__(self, hidden_layers_sizes, activation=tf.nn.relu):
        super().__init__()

        self.activation = activation
        self.net = tf.keras.models.Sequential()

        for layer_size in hidden_layers_sizes[:-1]:
            layer = tf.keras.layers.Dense(layer_size, activation=self.activation)
            self.net.add(layer)

        last_layer = tf.keras.layers.Dense(
            hidden_layers_sizes[-1], activation=tf.keras.activations.sigmoid
        )
        self.net.add(last_layer)

    def call(self, inputs):
        return self.net(inputs)

