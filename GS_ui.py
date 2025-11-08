import numpy as np
from sympy import Matrix
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
)
from GS_MatrixEditor import MatrixEditor
from PyQt5.QtGui import QFont
from config import STYLESHEET
from GS_MatrixDisplayWindow import MatrixDisplayWindow

class GeoSpanUI(QWidget): #继承QWIdget
    def __init__(self):
        super().__init__()
        self.editor = MatrixEditor()
        self.last_matrix = self.editor.get_matrix() #记录矩阵用于撤销
        self.init_ui()

    def init_ui(self):#设计窗口内容
        self.setWindowTitle("GeoSpan")
        self.resize(575, 643)
        font = QFont("Consolas", 10)
        self.setFont(font)
        #——————————————————————————————————————————————————————————————
        main_layout = QVBoxLayout() #总布局
        #——————————————————————————————————————————————————————————————
        maintitle_label = QLabel("GeoSpan v1.0 by Specptr")
        main_layout.addWidget(maintitle_label)
        #——————————————————————————————————————————————————————————————
        table_label = QLabel("Matrix:")
        main_layout.addWidget(table_label)
        #——————————————————————————————————————————————————————————————
        self.table = QTableWidget() #表格 用于置放矩阵
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents) #表格大小自定义
        main_layout.addWidget(self.table)
        #——————————————————————————————————————————————————————————————
        shape_layout = QHBoxLayout()

        shape_layout.addWidget(QLabel("Rows:"))
        self.row_input = QLineEdit("3")
        shape_layout.addWidget(self.row_input)

        shape_layout.addWidget(QLabel("Cols:"))
        self.col_input = QLineEdit("3")
        shape_layout.addWidget(self.col_input)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.update_matrix_shape)
        shape_layout.addWidget(confirm_btn)

        main_layout.addLayout(shape_layout)
        #——————————————————————————————————————————————————————————————
        op_row = QHBoxLayout()

        self.target_input = QLineEdit() #target row
        self.coeff1_input = QLineEdit() #coef 1
        self.divisor_input = QLineEdit() #divisor
        self.coeff2_input = QLineEdit() #coef 2
        self.src2_input = QLineEdit() #row 2

        op_row.addWidget(QLabel("Target Row"))
        op_row.addWidget(self.target_input)

        op_row.addWidget(QLabel("×"))
        op_row.addWidget(self.coeff1_input)

        op_row.addWidget(QLabel("/"))
        op_row.addWidget(self.divisor_input)

        op_row.addWidget(QLabel("+"))
        op_row.addWidget(self.coeff2_input)

        op_row.addWidget(QLabel("× Row"))
        op_row.addWidget(self.src2_input)

        main_layout.addLayout(op_row)
        #——————————————————————————————————————————————————————————————
        swap_layout = QHBoxLayout() #行交换布局

        swap_layout.addWidget(QLabel("Swap Row"))
        self.swap_a_input = QLineEdit()
        swap_layout.addWidget(self.swap_a_input)

        swap_layout.addWidget(QLabel("and Row"))
        self.swap_b_input = QLineEdit()
        swap_layout.addWidget(self.swap_b_input)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_combined_operation)
        apply_btn.setMinimumWidth(140)
        swap_layout.addWidget(apply_btn)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_last)
        undo_btn.setMinimumWidth(90)
        swap_layout.addWidget(undo_btn)

        main_layout.addLayout(swap_layout)
        #——————————————————————————————————————————————————————————————
        analysis_layout = QHBoxLayout()
        rref_btn = QPushButton("RREF")
        rref_btn.clicked.connect(self.perform_rref)

        null_btn = QPushButton("kernel")
        null_btn.clicked.connect(self.compute_nullspace)

        analysis_layout.addWidget(rref_btn)
        analysis_layout.addWidget(null_btn)
        main_layout.addLayout(analysis_layout)
        #——————————————————————————————————————————————————————————————
        self.setLayout(main_layout)
        self.update_matrix_shape()
        self.setStyleSheet(STYLESHEET)

    def update_table(self):#格式化表格，设置表格大小，控制元素居中，补齐0
        mat = self.editor.get_matrix()
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setRowCount(mat.shape[0])
        self.table.setColumnCount(mat.shape[1])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = "" if np.isnan(mat[i, j]) else str(int(mat[i, j])) if mat[i, j].is_integer() else str(mat[i, j])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)
        cell_size = 70
        for i in range(self.table.rowCount()):
            self.table.setRowHeight(i, cell_size)
            for j in range(self.table.columnCount()):
                self.table.setColumnWidth(j, cell_size)
                item = self.table.item(i, j)
                if not item:
                    item = QTableWidgetItem("0")
                    self.table.setItem(i, j, item)
                item.setTextAlignment(Qt.AlignCenter)

    def update_matrix_shape(self):#调整行数和列数
        try:
            rows = int(self.row_input.text())
            cols = int(self.col_input.text())
            old_rows = self.table.rowCount()
            old_cols = self.table.columnCount()
            new_mat = np.full((rows, cols), np.nan)
            for i in range(min(rows, old_rows)):
                for j in range(min(cols, old_cols)):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            new_mat[i, j] = float(item.text())
                        except ValueError:
                            pass
            self.editor.set_matrix(new_mat)
            self.last_matrix = new_mat.copy()
            self.update_table()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid row/col input")

    def apply_combined_operation(self):#实现行操作
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols), dtype=float)
            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except ValueError:
                            pass
            self.editor.set_matrix(mat)
            self.last_matrix = mat.copy()

            row_op_done = False
            target_text = self.target_input.text().strip()
            mul_text = self.coeff1_input.text().strip()
            div_text = self.divisor_input.text().strip()
            coeff2_text = self.coeff2_input.text().strip()
            src2_text = self.src2_input.text().strip()
            if target_text:  # 只要目标行存在就执行
                target = int(target_text)
                t = target - 1
                mul = float(mul_text) if mul_text else 1# 默认乘数和除数为 1，如果填写则使用填写值
                div = float(div_text) if div_text else 1
                scale = mul / div
                new_row = scale * mat[t]
                if src2_text:# 如果填写了第二行索引，则参与加权
                    src2 = int(src2_text) - 1
                    coeff2 = float(coeff2_text) if coeff2_text else 1#默认1
                    new_row += coeff2 * mat[src2]
                self.editor.matrix[t] = new_row
                row_op_done = True

            swap_done = False
            swap_a = self.swap_a_input.text().strip()
            swap_b = self.swap_b_input.text().strip()
            if swap_a and swap_b:
                a = int(swap_a) - 1
                b = int(swap_b) - 1
                temp = self.editor.matrix[a].copy()
                self.editor.matrix[a] = self.editor.matrix[b].copy()
                self.editor.matrix[b] = temp
                swap_done = True

            if not row_op_done and not swap_done:
                raise ValueError("Empty?")
            self.update_table()#清空所有输入框
            self.target_input.clear()
            self.coeff1_input.clear()
            self.divisor_input.clear()
            self.coeff2_input.clear()
            self.src2_input.clear()
            self.swap_a_input.clear()
            self.swap_b_input.clear()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def undo_last(self):#撤回
        self.editor.set_matrix(self.last_matrix)
        self.update_table()

    def perform_rref(self): #计算rref并显示
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols), dtype=float)
            for i in range(rows): #读取
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except ValueError:
                            pass
            sym_mat = Matrix(mat.tolist()) #numpy换sumpy的Matrix
            rref_mat, _ = sym_mat.rref()
            rref_np = rref_mat.tolist()
            self.rref_window = MatrixDisplayWindow(rref_np, title="RREF")
            self.rref_window.show()
        except Exception as e:
            QMessageBox.warning(self, "RREF Error", str(e))

    def compute_nullspace(self):#计算kernel并显示
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols), dtype=float)
            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except ValueError:
                            pass
            nullspace = Matrix(mat.tolist()).nullspace()
            vectors = [v.T.tolist()[0] for v in nullspace]
            self.null_window = MatrixDisplayWindow(vectors, title="Kernel", show_zero_if_empty=True)
            self.null_window.show()
        except Exception as e:
            QMessageBox.warning(self, "Null Space Error", str(e))
