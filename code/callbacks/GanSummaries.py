import tensorflow as tf
import numpy as np
import PIL.Image as Image
import io
import cv2

from .utils import array_to_image_summary
from .utils import maybe_to_rgb


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
        **kwargs
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
                    tag="Noise_to_Generated/{}".format(i),
                    image=array_to_image_summary(maybe_to_rgb(to_show)),
                )
            )
        summary = tf.Summary(value=summary_values)
        self.tensorboard_callback.writer.add_summary(summary, epoch)


class DiscriminatorGanSummary(tf.keras.callbacks.Callback):
    """Callback that adds image summaries for semantic segmentation to an existing
    tensorboard callback."""

    def __init__(
        self,
        tensorboard_callback,
        data,
        discriminator,
        disc_threshold=0.5,
        post_process_fn=lambda x: (x * 255).astype(np.uint8),
        update_freq=10,
        cmap="gray",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tensorboard_callback = tensorboard_callback
        self.data = data
        self.update_freq = update_freq
        self.post_process_fn = post_process_fn
        self.discriminator = discriminator
        self.disc_threshold=disc_threshold

        self.colormap = cmap

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_freq != 0:
            return
        summary_values = []

        preds = self.model.predict(self.data)

        for i, pred in enumerate(preds):

            to_show = self.post_process_fn(pred)
            to_show = maybe_to_rgb(to_show)

            is_real = self.discriminator.predict(pred[np.newaxis,...])[0][0]

            color = (255,0,0) if is_real < self.disc_threshold else (0,255,0)

            pad = 8
            to_show = cv2.copyMakeBorder(to_show,pad,2,2,2,cv2.BORDER_CONSTANT,value=color)
            to_show = cv2.putText(to_show, str(round(is_real,2)),(2,7),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

            summary_values.append(
                tf.Summary.Value(
                    tag="Noise_to_Generated/{}".format(i),
                    image=array_to_image_summary(to_show),
                )
            )
        summary = tf.Summary(value=summary_values)
        self.tensorboard_callback.writer.add_summary(summary, epoch)