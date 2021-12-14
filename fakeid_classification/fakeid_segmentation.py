"""FakeID segmentation module"""

import configparser
import logging

import cv2 as cv
import numpy as np
import tensorflow as tf

logger = logging.getLogger("fakeid_segmentation")


class FakeIDSegmentation:
    """FakeID segmentation class.

    Labels:
        0 --> Background
        1 --> ID Card
    """

    def __init__(self, modelpath):
        """Class instantiation.

        Parameters
        ----------
        modelpath : str
            Path to hdf5 model
        target_size : tuple, optional
            Image target size, by default (224, 224)
        threshold : float, optional
            Mask threshold. By default 0.5
        """

        self.model = tf.keras.models.load_model(modelpath, compile=False)

        config = configparser.ConfigParser()
        config.read("fakeid.ini")

        self.target_size = (
            config.getint("Segmentation", "SegmentationInputSizeX"),
            config.getint("Segmentation", "SegmentationInputSizeY"),
        )
        self.threshold = config.getfloat("Segmentation", "SegmentationThreshold")
        self.pad = config.getint("Segmentation", "SegmentationPad")

        logger.debug(f"Instanciating {self}\nModel path: '{modelpath}'")

    def forward(self, image):
        """Apply sequentially all model methods to get a final bitwised and cropped
        image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Cropped and bitwised image with [height, width, channels]
        """

        image_processed, original_shape = self.process_image(image)
        mask = self.predict(image_processed, original_shape)
        croped_im, croped_mask = self.crop(image, mask)
        bwed_image = self.bitwise_image(croped_im, croped_mask)

        return bwed_image

    def process_image(self, image):
        """Resample and process the input image. The image is first resampled using
        OpenCV with cv2.INTER_AREA interpolation for decimation, or cv2.INTER_CUBIC for
        expansion.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Reshaped and processed image.
        tuple
            Original size of input image.
        """
        original_shape = image.shape[:-1]

        if image.shape[:2] != self.target_size[::-1]:
            image = cv.resize(image, self.target_size)

        # normalize image
        image = image.astype(np.float32) / 255.0

        return image, original_shape

    def predict(self, image, original_shape=(None, None)):
        """Infer the mask of the input image. The image to be masked must first be
        processed using the process_image() function. The image is reshaped and
        normalized before inference.

        Parameters
        ----------
        image : numpy.ndarray
            Image to segment.
        original_shape : tuple
            Original image size, unknown by default (None, None)

        Returns
        -------
        numpy.ndarray
            Image mask with size equal to original size.
        """
        # import cv2
        # cv2.imshow('image', image)
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        pred_mask = self.model.predict(image)[0]
        pred_shape = pred_mask.shape

        if original_shape != (None, None):
            pred_mask = cv.resize(pred_mask, original_shape[::-1])

        # process mask
        if self.threshold == 0.0:
            ci_mask = np.argmax(pred_mask, axis=-1)
            logger.debug("Mask thresholded with numpy.argmax function")
        else:
            ci_mask = pred_mask[..., 1] >= self.threshold
            logger.debug(f"Mask thresholded with {self.threshold}")
        logger.debug(
            f"Predicted mask shape: {pred_shape}\nResized mask shape: {ci_mask.shape}"
        )

        return ci_mask.astype(np.uint8)

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.

        Parameters
        ----------
        mask: numpy.ndarray
            [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns
        -------
        numpy.ndarray
            bbox array [num_instances, (y1, x1, y2, x2)].
        """
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]

            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to resizing or cropping.
                # Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0

            boxes[i] = np.array([y1, x1, y2, x2])

        return boxes.astype(np.int32)

    def crop(self, image, mask):
        """Extract bounding box with a given mask to crop original image and mask.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        mask : numpy.ndarray
            Predicted mask with [height, width] format.

        Returns
        -------
        numpy.ndarray
            Cropped image with [height, width, channels] format.
        numpy.ndarray
            Cropped mask with [height, width] format.
        """
        # check integrity
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # get CI bbox
        y1, x1, y2, x2 = self.extract_bboxes(mask)[0]

        # add pads to bbox
        x1 -= self.pad
        y1 -= self.pad
        x2 += self.pad
        y2 += self.pad

        # correct pads
        if x1 <= 0:
            x1 = 1
        if y1 <= 0:
            y1 = 1
        if x2 >= mask.shape[1]:
            x2 = mask.shape[1] - 1
        if y2 >= mask.shape[0]:
            y2 = mask.shape[0] - 1

        # crop image and mask
        image = image[y1:y2, x1:x2, ...]  # multi channel crop
        mask = mask[y1:y2, x1:x2, -1]  # get unique instance of mask

        return image, mask

    def bitwise_image(self, image, mask):
        """Do bitwise operation to extract only image pixels given a mask.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        mask : numpy.ndarray
            Predicted mask with [height, width] format.

        Returns
        -------
        numpy.ndarray
            Bitwised image with [height, width, channels]
        """

        mask = cv.merge((mask, mask, mask))
        mask = (mask * 255).astype(np.uint8)
        image = mask & image  # or np.logical_and

        return image.astype(np.uint8)

    def __str__(self):
        return (
            f"MobileU-Net {type(self).__name__} network\n"
            f"Target size: {self.target_size}\n"
            f"Mask threshold: {self.threshold}\n"
            f"Pad: {self.pad}"
        )
