from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHBoxLayout
from config import STYLESHEET
import numpy as np

class MatrixDisplayWindow(QWidget) : # 对rref和kernel的窗口化显示
    def __init__(self, matrix, title="Result", show_zero_if_empty=False, event="None"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(575, 425)
        self.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout()
        table = QTableWidget() # 表格显示
        if show_zero_if_empty and not matrix: # 如果是空零空间，显示单个 0
            table.setRowCount(1)
            table.setColumnCount(1)
            item = QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, 0, item)
            table.setHorizontalHeaderLabels(["None"])
        else:
            table.setRowCount(len(matrix))
            table.setColumnCount(len(matrix[0]) if matrix else 0)

            # 设置表头：v1, v2, v3...
            if event == "kernel":
                if matrix and len(matrix[0]) > 0:
                    headers = [f"v{j+1}" for j in range(len(matrix[0]))]
                    table.setHorizontalHeaderLabels(headers)

            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    rounded = round(float(val), 2)
                    display = str(int(rounded)) if rounded.is_integer() else f"{rounded:.2f}".rstrip("0").rstrip(".")
                    item = QTableWidgetItem(display)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, j, item)

        if event == "rref":
            table.horizontalHeader().hide() # 隐藏表头
        table.verticalHeader().hide()


        cell_size = 80
        for i in range(table.rowCount()):
            table.setRowHeight(i, cell_size)
        for j in range(table.columnCount()):
            table.setColumnWidth(j, cell_size)
        layout.addWidget(table)
        self.setLayout(layout)

class EigenDisplayWindow(QWidget):
    def __init__(self, eigen_data):
        super().__init__()
        self.setWindowTitle("Eigenvectors")
        self.resize(650, 450)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout()
        table = QTableWidget()
        layout.addWidget(table)
        self.setLayout(layout)
        # 解析 eigen_data 格式 统计所有向量的最大维度 能多个 λ 维度相同
        all_vectors = []
        headers = []

        for (eig_val, _, vec_list) in eigen_data:
            for idx, vec in enumerate(vec_list):
                numeric_vec = np.array(vec, dtype=float).reshape(-1)
                all_vectors.append(numeric_vec)

                if float(eig_val).is_integer():
                    header_val = str(int(eig_val))
                else:
                    header_val = f"{eig_val:.1f}"

                headers.append(f"λ={header_val}")

        if not all_vectors:  # 无特征向量情况
            table.setRowCount(1)
            table.setColumnCount(1)
            table.setItem(0, 0, QTableWidgetItem("None"))
            table.horizontalHeader().hide()
            table.verticalHeader().hide()
            return

        max_dim = max(len(v) for v in all_vectors)

        table.setRowCount(max_dim)
        table.setColumnCount(len(all_vectors))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()

        for col, vec in enumerate(all_vectors):
            for row in range(max_dim):
                if row < len(vec):
                    val = float(vec[row])
                    display = str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip("0").rstrip(".")
                else:
                    display = ""

                item = QTableWidgetItem(display)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)

        cell_size = 80
        for r in range(max_dim):
            table.setRowHeight(r, cell_size)
        for c in range(len(all_vectors)):
            table.setColumnWidth(c, cell_size)

class DiagonalizeDisplayWindow(QWidget):
    def __init__(self, P, D):
        super().__init__()
        self.setWindowTitle("Diagonalization (P and D)")
        self.resize(900, 500)
        self.setStyleSheet(STYLESHEET)
        layout = QHBoxLayout()
        self.setLayout(layout)
        table_P = QTableWidget()
        self._fill_table(table_P, P, title="P Matrix")
        layout.addWidget(table_P)
        table_D = QTableWidget()
        self._fill_table(table_D, D, title="D Matrix")
        layout.addWidget(table_D)

    def _fill_table(self, table, mat, title=None):
        mat = np.array(mat, dtype=float)
        rows, cols = mat.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)

        headers = [title] + [""] * (cols - 1)
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()

        for i in range(rows):
            for j in range(cols):
                val = mat[i, j]
                display = str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip("0").rstrip(".")
                item = QTableWidgetItem(display)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, item)

        cell_size = 80
        for i in range(rows):
            table.setRowHeight(i, cell_size)
        for j in range(cols):
            table.setColumnWidth(j, cell_size)

class MessageWindow(QWidget):
    def __init__(self, title="Error"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(300, 300)
        self.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout()
        table = QTableWidget(1, 2)
        table.setFixedSize(400, 100)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setItem(0, 0, QTableWidgetItem("Cannot"))
        table.setItem(0, 1, QTableWidgetItem("compute"))
        table.item(0, 0).setTextAlignment(Qt.AlignCenter)
        table.item(0, 1).setTextAlignment(Qt.AlignCenter)
        layout.addWidget(table)
        self.setLayout(layout)
