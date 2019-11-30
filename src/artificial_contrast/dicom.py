import pydicom
import cv2
import numpy as np

from fastai.vision import Image, ImageSegment, pil2tensor


def read_HU_array(fn):
    if not isinstance(fn, str):
        fn = str(fn)
    dcm = pydicom.dcmread(fn)
    return dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept


def _normalize(array, window_min, window_max):
    return ((array - window_min) / (window_max - window_min) * (255 - 0) + 0).astype(
        np.uint8
    )


def open_dcm_image_factory(conf):
    def open_dcm_image(file_name, *args, **kwargs) -> Image:
        arr = read_HU_array(file_name)

        windowed_arrays = []
        for window_min, window_max in conf['windows']:
            array = np.clip(arr, a_min=window_min, a_max=window_max)
            array = _normalize(array, window_min, window_max)

            windowed_arrays.append(array)

        final_array = np.dstack(windowed_arrays)
        return Image(pil2tensor(final_array, np.float32).div_(255))

    return open_dcm_image_factory


def open_dcm_mask(path, *args, **kwargs) -> Image:
    array = pydicom.dcmread(path).pixel_array
    return ImageSegment(pil2tensor(array, np.float32))
