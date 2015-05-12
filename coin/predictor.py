from os import path as op

import numpy as np
import caffe


# Neural net base path
NN_PATH = op.join(op.dirname(op.realpath(__file__)), 'nn')

# Neural net configs
IMAGE_MEAN = op.join(NN_PATH, 'coinnet_e_large_mean.binaryproto')
DEPLOY = op.join(NN_PATH, 'deploy_coin_net.prototxt')
WEIGHTS = op.join(NN_PATH, 'large_cnn_train_iter_85000.caffemodel')


LABEL_TO_CLASS = {
    0: 'BAD',
    1: 'Q',
    2: 'Q',
    3: 'P',
    4: 'P',
    5: 'N',
    6: 'N',
    7: 'D',
    8: 'D',
}


def _load_image_mean():
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(IMAGE_MEAN, 'rb').read()
    blob.ParseFromString(data)
    return caffe.io.blobproto_to_array(blob)


net = caffe.Classifier(
    DEPLOY, WEIGHTS,
    raw_scale=255.0, input_scale=1.0,
    mean=_load_image_mean()[0, :, 14:241, 14:241],
)


def predict_bgr_uint8_images(image_list):
    """Returns a list of coin class predictions.

    :params: image_list is a list of bgr uint8 images.
    :return: a list of predictions
    """
    image_list = [im.astype('float') / 255.0 for im in image_list]
    results = net.predict(image_list)
    return [
        LABEL_TO_CLASS[np.argmax(results[idx, :])]
        for idx in range(results.shape[0])
    ]
