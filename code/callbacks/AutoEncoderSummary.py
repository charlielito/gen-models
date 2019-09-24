import tensorflow as tf
import numpy as np
import PIL.Image as Image
import io
import matplotlib.pyplot as plt


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

    def _make_image(self, array):
        """Converts an image array to the protobuf representation neeeded for image
        summaries."""
        height, width, channels = array.shape
        image = Image.fromarray(array)

        with io.BytesIO() as memf:
            image.save(memf, format="PNG")
            image_string = memf.getvalue()
        return tf.Summary.Image(
            height=height,
            width=width,
            colorspace=channels,
            encoded_image_string=image_string,
        )

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
                to_show = plot_to_array(fig)

                summary_values.append(
                    tf.Summary.Value(
                        tag=f"{split}+Input_Reconstructed/{i}",
                        image=self._make_image((to_show)),
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


def maybe_to_rgb(image):

    if image.ndim >= 3 and image.shape[2] == 1:
        image = np.stack([image[..., 0]] * 3, axis=-1)

    return image


def plot_to_array(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    with Image.open(buf) as img:
        image = np.asarray(img)[..., :3]

    return image

