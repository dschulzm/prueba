"""FakeID classification module"""

import configparser
import logging
from abc import ABC
import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2, resnet50, resnet
from scipy.spatial import distance

logger = logging.getLogger("fakeid_classification")


class FakeIDClassificationModel(ABC):
    """FakeID abstract base class."""

    def __init__(
        self,
        modelpath,
        target_size=(224, 224),
        bonafide_label=0,
        threshold=0.5,
        mask=None,
    ):
        """Class instantiation.

        Parameters
        ----------
        modelpath : str
            Path to h5 or saved model
        target_size : tuple, optional
            Image target size, by default (224, 224)
        bonafide_label : int, optional
            Indicates de label for the genuine class. By default 0
        threshold : float, optional
            Model threshold. By default 0.5
        mask : list, optional
            List of booleans to mask the genuine class when the threshold is not met.
            The numpy.argmax() function is used on the masked scores array to determine
            the correct prediction between attack presentation classes. By default None
        """
        self.model = tf.keras.models.load_model(modelpath, compile=False, custom_objects={'tf': tf})
        self.target_size = target_size
        self.bonafide_label = bonafide_label
        self.threshold = threshold
        self.mask = mask
        self.rgb_mean = np.zeros((1, 3))

        logger.debug(f"Instanciating {self}\nModel path: '{modelpath}'")

    def process_image(self, image, *preprocessing):
        """Resample and process the input image. The image is first resampled using
        OpenCV with cv2.INTER_AREA interpolation for decimation, or cv2.INTER_CUBIC for
        expansion.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        preprocessing : function, optional
            Any number of preprocessing functions as positional arguments. They can also
            be contained in a sequence or collection, but must be unpacked in the
            function call. Each function must receive a NumPy array as argument, and
            output a NumPy array of the same rank.

        Returns
        -------
        numpy.ndarray
            Resampled and processed image.
        """
        if image.shape[:2] > self.target_size[::-1]:
            image = cv.resize(image, self.target_size, interpolation=cv.INTER_AREA)
        elif image.shape[:2] < self.target_size[::-1]:
            image = cv.resize(image, self.target_size, interpolation=cv.INTER_CUBIC)
        for function in preprocessing:
            image = function(image)

        return image

    def classify(self, image):
        """Infer the class of the input image. The image to be classified must first be
        processed using the process_image() function. The image is reshaped and
        normalized before inference.

        Parameters
        ----------
        image : numpy.ndarray
            Image to classify in RGB format.

        Returns
        -------
        numpy.ndarray
            Confidence scores vector.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = self._normalize_features(image, -1, 1)

        scores = self.model.predict(image)[0]
        if scores[self.bonafide_label] >= self.threshold:
            prediction = self.bonafide_label
        else:
            masked = np.ma.masked_array(scores, mask=self.mask)
            prediction = np.argmax(masked)

        logger.debug(f"Prediction: {prediction} Scores vector: {scores}")

        return prediction, scores

    def _normalize_features(self, array, min_value, max_value):
        """Normalize the values of an array between min_value and max_value.

        Parameters
        ----------
        array : numpy.ndarray
            array to be normalized.
        min_value : int or float
            minimum value.
        max_value : int or float
            maximun value.

        Returns
        -------
        numpy.ndarray
            Array with normalized values.
        """
        return np.interp(
            array, (np.amin(array), np.amax(array)), (min_value, max_value)
        )

    def _mean_subtraction(self, image):
        """Subtract per-channel mean from image.

        Parameters
        ----------
        image : numpy.ndarray
            Original image.

        Returns
        -------
        numpy.ndarray
            Processed image of type np.float64.
        """
        return image.astype(np.float64) - self.rgb_mean

    def __str__(self):
        return (
            f"MobileNetV2 {type(self).__name__} network\n"
            f"Target size: {self.target_size}\n"
            f"Bona Fide label: {self.bonafide_label}\n"
            f"Threshold: {self.threshold}\n"
            f"Mask: {self.mask}\n"
            f"RGB mean: {self.rgb_mean}"
        )


class FakeID2(FakeIDClassificationModel):
    def __init__(self, modelpath):
        """Class instantiation.

        Parameters
        ----------
        modelpath : str
            Path to hdf5 model.
        """
        config = configparser.ConfigParser()
        config.read("fakeid.ini")

        target_size = (
            config.getint("FakeID2", "InputSizeX"),
            config.getint("FakeID2", "InputSizeY"),
        )
        threshold = config.getfloat("FakeID2", "Threshold")
        mask = [True, False, False]
        super().__init__(
            modelpath,
            target_size=target_size,
            bonafide_label=0,
            threshold=threshold,
            mask=mask,
        )

        self.templates = np.load(os.path.join(modelpath, 'templates.npy'))
        self.backbone = config.get("FakeID2", "Backbone")
        self.distance = config.get("FakeID2", "Distance")

        if self.backbone == 'mobilenetv2':
            self._preprocessor = mobilenet_v2.preprocess_input
        elif self.backbone == 'resnet50':
            self._preprocessor = resnet50.preprocess_input
        elif self.backbone == 'resnet101':
            self._preprocessor = resnet.preprocess_input
        elif self.backbone == 'resnet152':
            self._preprocessor = resnet.preprocess_input
        else:
            raise ValueError('Backbone %s is currently not supported' % self.backbone)

    def classify(self, image):
        """Infer the class of the input image. The image to be classified must first be
        processed using the process_image() function. The image is reshaped and
        normalized before inference.

        Parameters
        ----------
        image : numpy.ndarray
            Image to classify.

        Returns
        -------
        numpy.ndarray
            Confidence scores vector.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = self._preprocessor(image)

        features = self.model.predict(image)

        if self.distance == 'euclidean':
            dist = np.linalg.norm(features[0] - self.templates, axis=1)
        elif self.distance == 'cosine':
            dist = distance.cdist(features, self.templates, metric='cosine')
            if np.linalg.norm(features[0]) < 0.00000000001:  # Case when the CNN generates a vector that is all 0's
                dist = np.ones(dist.shape)
        else:  # Euclidean distance by default
            dist = np.linalg.norm(features[0] - self.templates, axis=1)
        avg_dist = dist.mean()

        score = 1 - np.clip(avg_dist, 0, 1)
        prediction = self.bonafide_label if score > self.threshold else self.bonafide_label + 1

        return prediction, score

    def process_image(self, image, *preprocessing):
        """Resample and process the input image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        preprocessing : function, optional
            Any number of preprocessing functions as positional arguments. They can also
            be contained in a sequence or collection, but must be unpacked in the
            function call. Each function must receive a NumPy array as argument, and
            output a NumPy array of the same rank.

        Returns
        -------
        numpy.ndarray
            Resampled and processed image.
        """
        if image.shape[:2] != self.target_size[::-1]:
            image = cv.resize(image, self.target_size)
        for function in preprocessing:
            image = function(image)

        return image
