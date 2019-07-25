import unittest
import numpy as np
import bayer_unify_aug

H, W = 4, 4


class MyTestCase(unittest.TestCase):

    def compare_mat(self, mat1, mat2):
        # print(mat1, "\n", mat2)
        np.testing.assert_array_equal(mat1, mat2)

    def test_bayer_unify_crop(self):
        raw4x4 = np.array(range(H * W)).reshape(H, W)
        raw4x2 = raw4x4[:, 1:-1]
        raw2x4 = raw4x4[1:-1, :]
        raw2x2 = raw4x4[1:-1, 1:-1]
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "GRBG", "crop"), raw4x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "GBRG", "crop"), raw4x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "RGGB", "crop"), raw4x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "BGGR", "crop"), raw4x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "GBRG", "crop"), raw2x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "GRBG", "crop"), raw2x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "BGGR", "crop"), raw2x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "RGGB", "crop"), raw2x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "BGGR", "crop"), raw2x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "RGGB", "crop"), raw2x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "GBRG", "crop"), raw2x2)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "GRBG", "crop"), raw2x2)

    def test_bayer_unify_pad(self):
        raw4x4 = np.array(range(H * W)).reshape(H, W)
        raw4x6 = np.concatenate([raw4x4[:, 1:2], raw4x4[:, :], raw4x4[:, -2:-1]], axis=1)
        raw6x4 = np.concatenate([raw4x4[1:2, :], raw4x4[:, :], raw4x4[-2:-1, :]], axis=0)
        raw6x6 = np.concatenate([raw6x4[:, 1:2], raw6x4[:, :], raw6x4[:, -2:-1]], axis=1)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "GRBG", "pad"), raw4x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "GBRG", "pad"), raw4x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "RGGB", "pad"), raw4x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "BGGR", "pad"), raw4x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "GBRG", "pad"), raw6x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "GRBG", "pad"), raw6x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "BGGR", "pad"), raw6x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "RGGB", "pad"), raw6x4)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "RGGB", "BGGR", "pad"), raw6x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "BGGR", "RGGB", "pad"), raw6x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GRBG", "GBRG", "pad"), raw6x6)
        self.compare_mat(bayer_unify_aug.bayer_unify(raw4x4, "GBRG", "GRBG", "pad"), raw6x6)

    def test_bayeraug(self):
        raw4x4 = np.array(range(H * W)).reshape(H, W)
        raw2x4 = raw4x4[H - 2:0:-1, :]
        raw4x2 = raw4x4[:, W - 2:0:-1]
        raw2x2 = raw4x4[H - 2:0:-1, W - 2:0:-1]

        # no transpose
        for pattern in bayer_unify_aug.BAYER_PATTERNS:
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, False, False, pattern), raw4x4)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, False, False, pattern), raw2x4)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, True, False, pattern), raw4x2)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, True, False, pattern), raw2x2)

        # transpose #1
        for pattern in ["RGGB", "BGGR"]:
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, False, True, pattern), raw4x4.T)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, False, True, pattern), raw2x4.T)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, True, True, pattern), raw4x2.T)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, True, True, pattern), raw2x2.T)

        # transpose #2
        rawt2x2 = raw4x4.T[1:-1, 1:-1]
        rawt2x4 = raw4x4[::-1, :].T[1:-1, :]
        rawt4x2 = raw4x4[:, ::-1].T[:, 1:-1]
        rawt4x4 = raw4x4[::-1, ::-1].T
        for pattern in ["GRBG", "GBRG"]:
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, False, True, pattern), rawt2x2)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, False, True, pattern), rawt2x4)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, False, True, True, pattern), rawt4x2)
            self.compare_mat(bayer_unify_aug.bayer_aug(raw4x4, True, True, True, pattern), rawt4x4)


if __name__ == '__main__':
    unittest.main()
