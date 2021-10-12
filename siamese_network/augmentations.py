import logging

import imgaug.augmenters as iaa

logger = logging.getLogger("fakeid.handler.augmentations")


def build_augmentation_sequence_light():
    # crop = iaa.Crop(px=(0, 16))
    blur = iaa.GaussianBlur(sigma=(0.0, 0.5))
    multiply = iaa.Multiply(mul=(0.7, 1.4))  # (0.5, 2.0))
    temperature = iaa.OneOf(
        [
            iaa.ChangeColorTemperature((4000, 8000)),
            iaa.ChangeColorTemperature((8000, 16000)),
            iaa.ChangeColorTemperature((20000, 40000)),
        ]
    )
    rotate_or_shear = iaa.OneOf(
        [iaa.Affine(rotate=(-5, 5), fit_output=False), iaa.Affine(shear=(-4, 4))]
    )
    return iaa.SomeOf(
        (3, None), [blur, multiply, temperature, rotate_or_shear], random_order=True
    )


def build_augmentation_sequence_medium(random_order=True):
    return iaa.SomeOf(
        (0, None),
        [  # None to all augmentations
            iaa.OneOf(
                [  # Affine
                    iaa.Affine(rotate=(-5, 5), order=4, mode="constant"),
                    iaa.Affine(shear=(-4, 4), order=4, mode="constant"),
                ]
            ),
            iaa.Multiply(mul=(0.7, 1.4)),  # Brightness
            iaa.OneOf(
                [  # Temperature
                    iaa.ChangeColorTemperature((4000, 8000)),  # Hot
                    iaa.ChangeColorTemperature((8000, 16000)),  # Warm
                    iaa.ChangeColorTemperature((20000, 40000)),  # Cold
                ]
            ),
            iaa.OneOf(
                [  # Blur
                    # iaa.GaussianBlur(sigma=(1, 3)),
                    # iaa.AverageBlur(k=(3, 5)),
                    iaa.MedianBlur(k=(3, 5)),
                    # iaa.imgcorruptlike.MotionBlur(severity=(1, 3)),
                    # iaa.imgcorruptlike.DefocusBlur(severity=(1, 2))
                ]
            ),
        ],
        random_order=random_order,
    )


def build_augmentation_sequence_heavy(
    affine_chance=0.5,
    crop_chance=0,  # 0.5,
    brightness_chance=0.5,
    temperature_chance=0.35,
    blur_chance=0.5,
    noise_chance=0.35,
    dropout_chance=0,  # 0.35,
    random_order=False,
):
    return iaa.SomeOf(
        (0, None),
        [  # None to all augmentations
            iaa.Sometimes(
                affine_chance,
                [  # Affine
                    iaa.SomeOf(
                        1,
                        [
                            iaa.Affine(rotate=(-25, 25), order=4, mode="constant"),
                            iaa.Affine(shear=(-16, 16), order=4, mode="constant"),
                            iaa.Sequential(
                                [  # Apply both but less agressive
                                    iaa.Affine(
                                        rotate=(-12, 12), order=4, mode="constant"
                                    ),
                                    iaa.Affine(shear=(-8, 8), order=4, mode="constant"),
                                ],
                                random_order=True,
                            ),
                        ],
                    )
                ],
            ),
            iaa.Sometimes(
                crop_chance,
                [iaa.Crop(percent=(0.05, 0.1), keep_size=False)],  # Zoom effect
            ),
            iaa.Sometimes(
                brightness_chance,
                [  # Brightness
                    iaa.SomeOf(
                        1,
                        [
                            iaa.Multiply(mul=(0.5, 1.5)),
                            iaa.Add((-50, 50)),
                            iaa.GammaContrast(gamma=(2, 0.5)),
                            # iaa.imgcorruptlike.Brightness(severity=(1, 2))
                        ],
                    )
                ],
            ),
            iaa.Sometimes(
                temperature_chance,
                [  # Temperature
                    iaa.SomeOf(
                        1,
                        [
                            iaa.ChangeColorTemperature(1000),  # Burning
                            iaa.ChangeColorTemperature((1000, 4000)),  # Hot
                            iaa.ChangeColorTemperature((4000, 10000)),  # Warm
                            iaa.ChangeColorTemperature((20000, 40000)),  # Cold
                            iaa.ChangeColorTemperature(40000),  # Very cold
                            iaa.Sequential(
                                [  # Freezing
                                    iaa.ChangeColorTemperature(40000),
                                    iaa.ChangeColorTemperature(40000),
                                ]
                            ),
                        ],
                    )
                ],
            ),
            iaa.Sometimes(
                blur_chance,
                [  # Blur
                    iaa.SomeOf(
                        1,
                        [
                            iaa.GaussianBlur(sigma=(1, 3)),
                            iaa.AverageBlur(k=(3, 5)),
                            iaa.MedianBlur(k=(3, 5)),
                            # iaa.imgcorruptlike.MotionBlur(severity=(1, 3)),
                            # iaa.imgcorruptlike.DefocusBlur(severity=(1, 2)),
                        ],
                    )
                ],
            ),
            iaa.Sometimes(
                noise_chance,
                [  # Noise
                    iaa.SomeOf(
                        1,
                        [
                            # iaa.AdditiveGaussianNoise(scale=(.01*255, .03*255)),
                            # iaa.AdditiveLaplaceNoise(scale=(.01*255, .03*255)),
                            # iaa.AdditivePoissonNoise(lam=(4, 12)),
                            iaa.imgcorruptlike.GaussianNoise(severity=1),
                            iaa.imgcorruptlike.ShotNoise(severity=1),
                            iaa.imgcorruptlike.SpeckleNoise(severity=1),
                        ],
                    )
                ],
            ),
            iaa.Sometimes(
                dropout_chance,
                [  # Drop areas
                    iaa.CoarseDropout(p=(0.1, 0.2)),
                ],
            ),
        ],
        random_order=random_order,
    )
