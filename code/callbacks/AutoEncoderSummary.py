import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .utils import plt_figure_to_array, array_to_image_summary

class AutoEncoderSummary(tf.keras.callbacks.Callback):
    """Callback that adds image summaries for semantic segmentation to an existing
    tensorboard callback."""

    def __init__(
        self, tensorboard_callback, data, update_freq=10, cmap="gray", **kwargs
    ):
        super().__init__(**kwargs)
        self.tensorboard_callback = tensorboard_callback
        self.data = data
        self.update_freq = update_freq

        self.colormap = cmap

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_freq != 0:
            return
        summary_values = []
        for inputs, split in self.data:

            preds = self.model.predict(inputs)

            for i, (image, pred) in enumerate(zip(inputs, preds)):

                to_show = (pred.reshape(28, 28, 1) * 255).astype(np.uint8)
                image = (image.reshape(28, 28, 1) * 255).astype(np.uint8)

                fig = self.get_figure_images(
                    image.reshape(28, 28), pred.reshape(28, 28)
                )
                to_show = plt_figure_to_array(fig)

                summary_values.append(
                    tf.Summary.Value(
                        tag="{}+Input_Reconstructed/{}".format(split,i),
                        image=array_to_image_summary(to_show),
                    )
                )
        summary = tf.Summary(value=summary_values)
        self.tensorboard_callback.writer.add_summary(summary, epoch)

    def get_figure_images(self, image1, image2):
        figure = plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("original", fontsize=30)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image1, cmap=self.colormap)
        plt.subplot(1, 2, 2)
        plt.title("autoencoder", fontsize=30)
        plt.imshow(image2, cmap=self.colormap)
        plt.xticks([])
        plt.yticks([])

        return figure

