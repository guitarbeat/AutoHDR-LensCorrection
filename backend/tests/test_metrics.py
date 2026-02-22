import unittest
import numpy as np
from backend.evaluation.metrics import calculate_ssim

class TestMetrics(unittest.TestCase):
    def test_calculate_ssim_identical(self):
        img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        ssim = calculate_ssim(img, img)
        self.assertAlmostEqual(ssim, 1.0, places=4)

    def test_calculate_ssim_different(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        ssim = calculate_ssim(img1, img2)
        # SSIM should be low (close to 0 or negative?)
        # With global means 0 and 255, it might be small but positive due to constants.
        self.assertTrue(0 <= ssim < 1.0)

    def test_calculate_ssim_shape_mismatch(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((50, 50, 3), dtype=np.uint8)
        # Should resize img2 to match img1
        ssim = calculate_ssim(img1, img2)
        self.assertAlmostEqual(ssim, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
