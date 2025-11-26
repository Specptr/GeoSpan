import numpy as np

class MatrixEditor:
    def __init__(self):
        self.matrix = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]], dtype=float)
        self.rows, self.cols = self.matrix.shape

    def update_shape(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.matrix = np.full((rows, cols), np.nan)

    def set_element(self, i, j, value):
        self.matrix[i, j] = value

    def get_matrix(self):
        return self.matrix.copy()

    def set_matrix(self, new_matrix):
        self.matrix = new_matrix.copy()

    def multiply_vector(self, vec):
        mat = self.matrix
        if mat.shape[1] != vec.shape[0]:
            raise ValueError("Row inequal")
        return np.dot(mat, vec)
