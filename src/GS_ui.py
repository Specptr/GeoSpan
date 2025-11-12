import numpy as np
from sympy import Matrix
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox
)
from GS_MatrixEditor import MatrixEditor
from PyQt5.QtGui import QFont
from config import STYLESHEET
from GS_MatrixDisplayWindow import MatrixDisplayWindow

class GeoSpanUI(QWidget): # 窗口 运算逻辑 日志
    def __init__(self):
        super().__init__()
        self.editor = MatrixEditor() # 记录矩阵为ndarray 与表格交互
        self.last_matrix = self.editor.get_matrix() #记录矩阵用于撤销
        self.sys_operation_count = -1 # 系统操作步数 抵消初始化为3x3的1步操作
        self.sys_operation_log_lines = [] # 系统操作日志
        self.row_operation_count = 0 # 行操作日志步数
        self.row_operation_log_lines = [] # 行操作日志
        self.undo_stack = []  # 存储撤销的操作（用于 redo）
        self.operation_history = []  # 存储每一步的矩阵快照
        self.init_ui()

    def init_ui(self):#设计窗口内容
        self.setWindowTitle("GeoSpan")
        self.resize(500, 1055)
        font = QFont("Consolas", 10)
        self.setFont(font)
        #——————————————————————————————————————————————————————————————
        main_layout = QVBoxLayout() # 总布局
        #——————————————————————————————————————————————————————————————
        maintitle_label = QLabel("GeoSpan v1.1 by EnoLaice")
        main_layout.addWidget(maintitle_label)
        #——————————————————————————————————————————————————————————————
        table_label = QLabel("Matrix:")
        main_layout.addWidget(table_label)
        #——————————————————————————————————————————————————————————————
        self.table = QTableWidget() # 表格 用于置放矩阵
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents) # 表格大小自定义
        main_layout.addWidget(self.table)
        #——————————————————————————————————————————————————————————————
        shape_layout = QHBoxLayout()

        shape_layout.addWidget(QLabel("Rows:"))
        self.row_input = QLineEdit("3")
        shape_layout.addWidget(self.row_input)

        shape_layout.addWidget(QLabel("Cols:"))
        self.col_input = QLineEdit("3")
        shape_layout.addWidget(self.col_input)

        reshape_btn = QPushButton("Reshape")
        reshape_btn.clicked.connect(self.reshape_matrix)
        shape_layout.addWidget(reshape_btn)

        main_layout.addLayout(shape_layout)
        #——————————————————————————————————————————————————————————————
        init_layout = QHBoxLayout()

        zero_btn = QPushButton("Clear")
        zero_btn.clicked.connect(self.clear_matrix)
        init_layout.addWidget(zero_btn)

        rand_btn = QPushButton("Random Fill")
        rand_btn.clicked.connect(self.random_fill_matrix)
        init_layout.addWidget(rand_btn)

        main_layout.addLayout(init_layout)
        #——————————————————————————————————————————————————————————————
        op_row1 = QHBoxLayout()

        self.target_input = QLineEdit()
        self.coeff1_input = QLineEdit()
        self.divisor_input = QLineEdit()
        self.coeff2_input = QLineEdit()
        self.src2_input = QLineEdit() # 读取输入

        op_row1.addWidget(QLabel("Target Row"))
        op_row1.addWidget(self.target_input)

        op_row1.addWidget(QLabel("×"))
        op_row1.addWidget(self.coeff1_input)

        op_row1.addWidget(QLabel("/"))
        op_row1.addWidget(self.divisor_input)

        op_row2 = QHBoxLayout()

        op_row2.addWidget(QLabel("                "))

        op_row2.addWidget(QLabel("+"))
        op_row2.addWidget(self.coeff2_input)

        op_row2.addWidget(QLabel("× Row"))
        op_row2.addWidget(self.src2_input)

        main_layout.addLayout(op_row1)
        main_layout.addLayout(op_row2)
        #——————————————————————————————————————————————————————————————
        swap_layout = QHBoxLayout()

        swap_layout.addWidget(QLabel("Swap Row"))
        self.swap_a_input = QLineEdit()
        swap_layout.addWidget(self.swap_a_input)

        swap_layout.addWidget(QLabel("and Row"))
        self.swap_b_input = QLineEdit()
        swap_layout.addWidget(self.swap_b_input)

        main_layout.addLayout(swap_layout)
        #——————————————————————————————————————————————————————————————
        apply_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_combined_operation)
        apply_layout.addWidget(apply_btn)

        main_layout.addLayout(apply_layout)
        #——————————————————————————————————————————————————————————————
        row_log_layout = QHBoxLayout()

        self.row_operation_log_text = QTextEdit()
        self.row_operation_log_text.setReadOnly(True)
        self.row_operation_log_text.setPlainText("<Row Operation>")
        row_log_layout.addWidget(self.row_operation_log_text)
        self.row_operation_log_text.setFixedHeight(140)

        main_layout.addLayout(row_log_layout)
        #——————————————————————————————————————————————————————————————
        sys_log_layout = QHBoxLayout()

        self.sys_operation_log_text = QTextEdit()
        self.sys_operation_log_text.setReadOnly(True)
        self.sys_operation_log_text.setPlainText("<System Log>")
        sys_log_layout.addWidget(self.sys_operation_log_text)
        self.sys_operation_log_text.setFixedHeight(140)

        main_layout.addLayout(sys_log_layout)
        #——————————————————————————————————————————————————————————————
        unredo_layout = QHBoxLayout()

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_last)
        unredo_layout.addWidget(undo_btn)

        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self.redo_last)
        unredo_layout.addWidget(redo_btn)

        main_layout.addLayout(unredo_layout)
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
        self.reshape_matrix()
        self.setStyleSheet(STYLESHEET)

    def update_table(self): # 格式化表格，设置表格大小，控制元素居中，补齐0
        mat = self.editor.get_matrix()
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide() # 隐藏表头
        self.table.setRowCount(mat.shape[0])
        self.table.setColumnCount(mat.shape[1])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                # 转换字符串为整数或浮点数
                value = " " if np.isnan(mat[i, j]) else str(int(mat[i, j])) if mat[i, j].is_integer() else str(mat[i, j])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item) # 填入表格
        cell_size = 65
        for i in range(self.table.rowCount()):
            self.table.setRowHeight(i, cell_size)
            for j in range(self.table.columnCount()):
                self.table.setColumnWidth(j, cell_size) # 控制表格形状为正方形
                item = self.table.item(i, j)
                if not item:
                    item = QTableWidgetItem("0")
                    self.table.setItem(i, j, item)
                item.setTextAlignment(Qt.AlignCenter)

    def reshape_matrix(self): # 修改矩阵大小
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
                            pass  # 忽略无法转换的内容
            self.editor.set_matrix(new_mat)
            self.last_matrix = new_mat.copy()
            self.update_table()
            self.sys_operation_count += 1 # 加入日志
            log_text = f"[{self.sys_operation_count}] Reshape to [{rows}, {cols}]"
            self.sys_operation_log_text.append(log_text)
            self.sys_operation_log_lines.append(log_text)
            self.operation_history.append({
                "matrix": new_mat.copy(),
                "op_type": "reshape",
                "log": log_text
            })
            self.undo_stack.clear()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid row/col input")

    def clear_matrix(self): # 清除至全零
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        blank_mat = np.full((rows, cols), np.nan)
        self.editor.set_matrix(blank_mat)
        self.update_table()
        self.sys_operation_count += 1
        self.undo_stack.clear()
        log_text = f"[{self.sys_operation_count}] Clear matrix"
        self.operation_history.append({
                "matrix": blank_mat.copy(),
                "op_type": "clear",
                "log": log_text
            })
        self.sys_operation_log_text.append(log_text)
        self.sys_operation_log_lines.append(log_text)

    def random_fill_matrix(self): # 填满随机1-9整数
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        rand_mat = np.random.randint(1, 10, size=(rows, cols)).astype(float)
        self.editor.set_matrix(rand_mat)
        self.update_table()
        self.sys_operation_count += 1
        log_text = f"[{self.sys_operation_count}] Randomly fill the matrix"
        self.sys_operation_log_text.append(log_text)
        self.sys_operation_log_lines.append(log_text)
        self.operation_history.append({
                "matrix": rand_mat.copy(),
                "op_type": "random",
                "log": log_text
            })
        self.undo_stack.clear()

    def apply_combined_operation(self): # 实现行操作
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols), dtype=float)
            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j) # 读取表格元素
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except ValueError:
                            pass
            self.editor.set_matrix(mat)
            self.last_matrix = mat.copy()

            row_op_done = False # 运算操作
            target_text = self.target_input.text().strip()
            mul_text = self.coeff1_input.text().strip()
            div_text = self.divisor_input.text().strip()
            coeff2_text = self.coeff2_input.text().strip()
            src2_text = self.src2_input.text().strip()
            if target_text:  # 只要目标行存在就执行
                target = int(target_text)
                t = target - 1
                mul = float(mul_text) if mul_text else 1 # 默认乘数和除数为 1，如果填写则使用填写值
                div = float(div_text) if div_text else 1
                scale = mul / div
                new_row = scale * mat[t]
                if src2_text: # 如果填写了第二行索引，则参与加权
                    src2 = int(src2_text) - 1
                    coeff2 = float(coeff2_text) if coeff2_text else 1 # 默认1
                    new_row += coeff2 * mat[src2]
                self.editor.matrix[t] = new_row
                row_op_done = True

            swap_done = False # 交换行操作
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

            self.update_table() # 清空所有输入框
            self.target_input.clear()
            self.coeff1_input.clear()
            self.divisor_input.clear()
            self.coeff2_input.clear()
            self.src2_input.clear()
            self.swap_a_input.clear()
            self.swap_b_input.clear()

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

        operation_str = "" # 行操作表达式

        if target_text:
            t = int(target_text)
            mul = float(mul_text) if mul_text else 1.0
            div = float(div_text) if div_text else 1.0

            base = f"Row({t})"
            scale_parts = []
            if mul != 1.0:
                scale_parts.append(f"{mul:.3g}{base}")
            else:
                scale_parts.append(base)
            if div != 1.0:
                scale_parts[-1] += f" / {div:.3g}"

            operation_str = f"{base} → {scale_parts[-1]}" # 第一部分 只对目标行做乘除

            if src2_text:
                src2 = int(src2_text)
                coeff2 = float(coeff2_text) if coeff2_text else 1.0
                sign = "+" if coeff2 >= 0 else "-"
                coeff2_abs = abs(coeff2)

                if coeff2_abs == 1.0:
                    operation_str += f" {sign} Row({src2})"
                else:
                    operation_str += f" {sign} {coeff2_abs:.3g}Row({src2})" # 第二部分 有其他行参与运算

        if swap_a and swap_b:
            a = int(swap_a)
            b = int(swap_b)
            operation_str = f"Swap Row({a}) ↔ Row({b})"

        if operation_str:
            self.sys_operation_count += 1
            self.row_operation_count += 1
            row_numbered_str = f"[{self.row_operation_count}] {operation_str}"
            sys_numbered_str = f"[{self.sys_operation_count}] {operation_str}"
            self.sys_operation_log_text.append(sys_numbered_str)
            self.sys_operation_log_lines.append(sys_numbered_str)
            self.row_operation_log_text.append(row_numbered_str)
            self.row_operation_log_lines.append(row_numbered_str)
            self.operation_history.append({
                "matrix": self.editor.matrix.copy(),
                "op_type": "row_op",
                "log": sys_numbered_str
            })
            self.undo_stack.clear()

    def undo_last(self): # 撤回
        if self.operation_history:
            op = self.operation_history.pop()
            self.undo_stack.append(op)  # 保存编号和快照
            self.sys_operation_count -= 1

            if self.operation_history:
                self.editor.set_matrix(self.operation_history[-1]["matrix"].copy()) # 恢复上一步矩阵
            else:
                rows = self.table.rowCount()
                cols = self.table.columnCount()
                self.editor.set_matrix(np.zeros((rows, cols)))

            self.update_table()
            self.sys_operation_log_text.append(f"↩ Undo [{self.sys_operation_count + 1}]")

            self.rebuild_row_log() # 重新生成行操作步骤
        else:
            QMessageBox.information(self, "Error", "No more operations")

    def redo_last(self):
        if self.undo_stack:
            op = self.undo_stack.pop()

            self.editor.set_matrix(op["matrix"].copy())
            self.operation_history.append(op)

            self.sys_operation_count += 1
            self.update_table()
            self.sys_operation_log_text.append(f"↪ Redo [{self.sys_operation_count}]")

            self.rebuild_row_log()
        else:
            QMessageBox.information(self, "Error", "No more operations")

    def rebuild_row_log(self): # 重建行操作日志
        self.row_operation_log_text.clear()
        self.row_operation_log_lines.clear()
        self.row_operation_count = 0
        self.row_operation_log_text.setPlainText("<Row Operation>")
        for op in self.operation_history:
            if op["op_type"] in ("row_op"):
                self.row_operation_count += 1
                numbered = f"[{self.row_operation_count}] {op['log'].split(']')[1].strip()}"
                self.row_operation_log_text.append(numbered)
                self.row_operation_log_lines.append(numbered)

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
