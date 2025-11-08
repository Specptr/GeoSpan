import numpy as np

class MatrixEditor:
    def __init__(self, rows=3, cols=3):
        self.update_shape(rows, cols)

    def update_shape(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.matrix = np.full((rows, cols), np.nan)

    def set_element(self, i, j, value):
        self.matrix[i, j] = value

    def get_matrix(self):
        return self.matrix.copy()

    def set_matrix(self, new_matrix):
        self.matrix = new_matrix.copy()