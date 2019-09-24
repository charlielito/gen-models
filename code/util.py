# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future

import os
import requests
import zipfile
import numpy as np
import pandas as pd
from scipy.misc import imread, imsave, imresize
from glob import glob
from tqdm import tqdm

from sklearn.utils import shuffle


def relu(x):
    return x * (x > 0)


def error_rate(p, t):
    return np.mean(p != t)


def getKaggleMNIST(path="../large_files/train.csv"):
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv(path).values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest


def init_weights(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)


def get_celeb(limit=None):
    if not os.path.exists("../large_files"):
        os.mkdir("../large_files")

    # eventual place where our final data will reside
    if not os.path.exists("../large_files/img_align_celeba-cropped"):

        # check for original data
        if not os.path.exists("../large_files/img_align_celeba"):
            # download the file and place it here
            if not os.path.exists("../large_files/img_align_celeba.zip"):
                print("Downloading img_align_celeba.zip...")
                download_file(
                    "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                    "../large_files/img_align_celeba.zip",
                )

            # unzip the file
            print("Extracting img_align_celeba.zip...")
            with zipfile.ZipFile("../large_files/img_align_celeba.zip") as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall("../large_files")

        # load in the original images
        filenames = glob("../large_files/img_align_celeba/*.jpg")
        N = len(filenames)
        print("Found %d files!" % N)

        # crop the images to 64x64
        os.mkdir("../large_files/img_align_celeba-cropped")
        print("Cropping images, please wait...")

        for i in range(N):
            crop_and_resave(filenames[i], "../large_files/img_align_celeba-cropped")
            if i % 1000 == 0:
                print("%d/%d" % (i, N))

    # make sure to return the cropped version
    filenames = glob("../large_files/img_align_celeba-cropped/*.jpg")
    return filenames


def crop_and_resave(inputfile, outputdir):
    # theoretically, we could try to find the face
    # but let's be lazy
    # we assume that the middle 108 pixels will contain the face
    im = imread(inputfile)
    height, width, color = im.shape
    edge_h = int(round((height - 108) / 2.0))
    edge_w = int(round((width - 108) / 2.0))

    cropped = im[edge_h : (edge_h + 108), edge_w : (edge_w + 108)]
    small = imresize(cropped, (64, 64))

    filename = inputfile.split("/")[-1]
    imsave("%s/%s" % (outputdir, filename), small)


def scale_image(im):
    # scale to (-1, +1)
    return (im / 255.0) * 2 - 1


def files2images_theano(filenames):
    # theano wants images to be of shape (C, D, D)
    # tensorflow wants (D, D, C) which is what scipy imread
    # uses by default
    return [scale_image(imread(fn).transpose((2, 0, 1))) for fn in filenames]


def files2images(filenames):
    return [scale_image(imread(fn)) for fn in filenames]


# functions for downloading file from google drive
def save_response_content(r, dest):
    # unfortunately content-length is not provided in header
    total_iters = 1409659  # in KB
    print("Note: units are in KB, e.g. KKB = MB")
    # because we are reading 1024 bytes at a time, hence
    # 1KB == 1 "unit" for tqdm
    with open(dest, "wb") as f:
        for chunk in tqdm(
            r.iter_content(1024), total=total_iters, unit="KB", unit_scale=True
        ):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def download_file(file_id, dest):
    drive_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(drive_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(drive_url, params=params, stream=True)

    save_response_content(response, dest)
