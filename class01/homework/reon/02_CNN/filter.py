import numpy as np

class Filter :
    def __init__(self):
        self.edge = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        self.identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.ridge = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        self.edge_detection = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.box_blur = np.array([[1, 1, 1,], [1, 1, 1], [1, 1, 1]]) / 9
        self.gaussian_blur_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        self.gaussian_blur_5 = np.array([[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4 ,6, 4, 1]]) / 256

    def handle(self, var_name):
        return getattr(self, var_name, 'Unknown')
