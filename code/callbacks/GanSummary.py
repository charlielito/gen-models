import tensorflow as tf
import numpy as np
import PIL.Image as Image
import io


class GanSummary(tf.keras.callbacks.Callback):
    """Callback that adds image summaries for semantic segmentation to an existing
    tensorboard callback."""

    def __init__(
        self,
        tensorboard_callback,
        data,
        post_process_fn=lambda x: (x * 255).astype(np.uint8),
        update_freq=10,
        cmap="gray",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tensorboard_callback = tensorboard_callback
        self.data = data
        self.update_freq = update_freq
        self.post_process_fn = post_process_fn

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

        preds = self.model.predict(self.data)

        for i, pred in enumerate(preds):

            to_show = self.post_process_fn(pred)

            summary_values.append(
                tf.Summary.Value(
                    tag=f"Noise_to_Generated/{i}",
                    image=self._make_image(maybe_to_rgb(to_show)),
                )
            )
        summary = tf.Summary(value=summary_values)
        self.tensorboard_callback.writer.add_summary(summary, epoch)


def maybe_to_rgb(image):

    if image.ndim >= 3 and image.shape[2] == 1:
        image = np.stack([image[..., 0]] * 3, axis=-1)

    return image

