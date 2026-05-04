import cv2 as cv
from sulllam.localization.extraction.base_extractor import BaseExtractor


class ORBConfigs:
    pass


class ORBFeatureExtractor(BaseExtractor):

    def __init__(self):
        super().__init__(name="ORB")
        self.orb = cv.ORB_create()

    def _extract(self, image):
        kp = self.orb.detect(image, None)
        kp, des = self.orb.compute(image, kp)
        if des is None:
            return ([], [])
        return (kp, des)
