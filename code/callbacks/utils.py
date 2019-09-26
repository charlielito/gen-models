import tensorflow as tf
import numpy as np
import PIL.Image as Image
import io
import matplotlib.pyplot as plt

def array_to_image_summary(array, format="png"):
    """Converts an image array to the protobuf representation neeeded for image
    summaries."""
    height, width, channels = array.shape
    image = Image.fromarray(array)

    with io.BytesIO() as memf:
        image.save(memf, format=format)
        image_string = memf.getvalue()
    return tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channels,
        encoded_image_string=image_string,
    )

def plt_figure_to_array(figure):
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


def maybe_to_rgb(image):

    if image.ndim >= 3 and image.shape[2] == 1:
        image = np.stack([image[..., 0]] * 3, axis=-1)

    return image

class Callbacks:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __getattr__(self, name):
        def method(*args, **kwargs):
            for callback in self.callbacks:
                getattr(callback, name)(*args, **kwargs)

        return method