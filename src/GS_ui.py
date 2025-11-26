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
from GS_MatrixDisplayWindow import MatrixDisplayWindow, EigenDisplayWindow, DiagonalizeDisplayWindow, MessageWindow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

    def init_ui(self):
        self.setWindowTitle("GeoSpan")
        self.resize(900, 1000)
        font = QFont("Consolas", 10)
        self.setFont(font)
        # ======================================================
        # 主布局
        main_layout = QVBoxLayout()

        hint_layout = QHBoxLayout()
        MatrixA_label = QLabel("|    Matrix A    |")
        VectorV_label = QLabel("| Vector v | Vector Av |")
        VectorV_label.setAlignment(Qt.AlignRight)
        hint_layout.addWidget(MatrixA_label)
        hint_layout.addWidget(VectorV_label)
        main_layout.addLayout(hint_layout)
        # ======================================================
        # 第一行：五等份布局
        top_layout = QHBoxLayout()

        # 主矩阵表格（占前三份）
        self.table = QTableWidget()
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        top_layout.addWidget(self.table)

        # 可填数的列向量表格（占一份）
        self.input_vector = QTableWidget()
        self.input_vector.setColumnCount(1)
        self.input_vector.setFixedWidth(125)
        top_layout.addWidget(self.input_vector, stretch=1)

        # 矩阵变换后的列向量表格（占一份，不可编辑）
        self.output_vector = QTableWidget()
        self.output_vector.setColumnCount(1)
        self.output_vector.setFixedWidth(125)
        self.output_vector.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑
        top_layout.addWidget(self.output_vector, stretch=1)

        # 把第一行加到主布局
        main_layout.addLayout(top_layout)
        #水印
        maintitle_label = QLabel("◇▪ ▫▪ ▫▪ ▪ ▫◆     GeoSpan v1.3 by EnoLaice     ◇▪ ▫▪ ▫▪ ▫▪ ▫◆\n")
        maintitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(maintitle_label)

        # 第二行：左右两栏
        bottom_layout = QHBoxLayout()  # 第二层主容器
        left_layout = QVBoxLayout()    # 左栏
        right_layout = QVBoxLayout()   # 右栏
        # ======================================================
        # 左栏
        # 尺寸控制区
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
        left_layout.addLayout(shape_layout)

        # 初始化按钮区
        init_layout = QHBoxLayout()
        rand_btn = QPushButton("Randomize")
        rand_btn.clicked.connect(self.random_fill_matrix)
        init_layout.addWidget(rand_btn)
        zero_btn = QPushButton("Clear")
        zero_btn.clicked.connect(self.clear_matrix)
        init_layout.addWidget(zero_btn)
        left_layout.addLayout(init_layout)

        # 行运算输入区
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
        op_row2.addWidget(QLabel(" "))
        op_row2.addWidget(QLabel("+"))
        op_row2.addWidget(self.coeff2_input)
        op_row2.addWidget(QLabel("× Row"))
        op_row2.addWidget(self.src2_input)

        left_layout.addLayout(op_row1)
        left_layout.addLayout(op_row2)

        # 交换行
        swap_layout = QHBoxLayout()
        self.swap_a_input = QLineEdit()
        self.swap_b_input = QLineEdit()
        swap_layout.addWidget(QLabel("Swap Row"))
        swap_layout.addWidget(self.swap_a_input)
        swap_layout.addWidget(QLabel("↔"))
        swap_layout.addWidget(self.swap_b_input)
        left_layout.addLayout(swap_layout)

        # 应用按钮
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_combined_operation)
        left_layout.addWidget(apply_btn)

        # 日志区
        # Row Operation 日志
        row_log_layout = QHBoxLayout()
        self.row_operation_log_text = QTextEdit()
        self.row_operation_log_text.setReadOnly(True)
        self.row_operation_log_text.setPlainText("<Row Operation>")
        self.row_operation_log_text.setFixedHeight(100)
        row_log_layout.addWidget(self.row_operation_log_text)
        left_layout.addLayout(row_log_layout)

        # System Log 日志
        sys_log_layout = QHBoxLayout()
        self.sys_operation_log_text = QTextEdit()
        self.sys_operation_log_text.setReadOnly(True)
        self.sys_operation_log_text.setPlainText("<System Log>")
        self.sys_operation_log_text.setFixedHeight(100)
        sys_log_layout.addWidget(self.sys_operation_log_text)
        left_layout.addLayout(sys_log_layout)

        # 撤销重做
        unredo_layout = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_last)
        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self.redo_last)
        unredo_layout.addWidget(undo_btn)
        unredo_layout.addWidget(redo_btn)
        left_layout.addLayout(unredo_layout)

        analysis_layout = QHBoxLayout()
        rref_btn = QPushButton("RREF")
        rref_btn.clicked.connect(self.perform_rref)
        null_btn = QPushButton("Kernel")
        null_btn.clicked.connect(self.compute_nullspace)
        analysis_layout.addWidget(rref_btn)
        analysis_layout.addWidget(null_btn)
        left_layout.addLayout(analysis_layout)
        # ======================================================
        # 右栏
        matrix_vector_calc_button = QPushButton("Do Linear Map")
        matrix_vector_calc_button.clicked.connect(self.update_output_vector)
        right_layout.addWidget(matrix_vector_calc_button)

        plot_btn = QPushButton("Plot 3D Matrix")
        plot_btn.clicked.connect(self.plot_3d_bar)
        right_layout.addWidget(plot_btn)

        self.figure = Figure(figsize=(5, 4), facecolor="#111111")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        right_layout.addWidget(self.canvas)

        eigen_layout = QHBoxLayout()
        eigenvector_btn = QPushButton("Eigenvectors")
        eigenvector_btn.clicked.connect(self.compute_eigenvectors)
        diagonalize_btn = QPushButton("Diagonalize")
        diagonalize_btn.clicked.connect(self.compute_diagonalize)
        eigen_layout.addWidget(eigenvector_btn)
        eigen_layout.addWidget(diagonalize_btn)
        right_layout.addLayout(eigen_layout)
        # ======================================================
        # 将左右布局加入第二行
        bottom_layout.addLayout(left_layout, 1)
        bottom_layout.addLayout(right_layout, 1)
        # 第二行加入主布局
        main_layout.addLayout(bottom_layout)

        # 应用整体样式
        self.setLayout(main_layout)
        self.update_table()
        self.init_default_vector()
        self.update_output_vector()   # 初始化计算一次
        self.reshape_matrix()
        self.setStyleSheet(STYLESHEET)
        self.plot_3d_bar()

    def update_table(self):
        mat = self.editor.get_matrix()
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()  # 隐藏表头
        self.table.setRowCount(mat.shape[0])
        self.table.setColumnCount(mat.shape[1])

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                # 如果值是 NaN，逻辑上设置为0，显示为空格
                if np.isnan(mat[i, j]):
                    display_text = " "
                    mat[i, j] = 0.0
                else:
                    val = mat[i, j]
                    # 显示整数或浮点数
                    display_text = str(int(val)) if val.is_integer() else str(val)

                item = QTableWidgetItem(display_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

        # 设置单元格大小为正方形
        cell_size = 65
        for i in range(self.table.rowCount()):
            self.table.setRowHeight(i, cell_size)
            for j in range(self.table.columnCount()):
                self.table.setColumnWidth(j, cell_size)
                item = self.table.item(i, j)
                if not item:
                    # 未创建单元格，也当作 0 逻辑值，显示空格
                    item = QTableWidgetItem(" ")
                    mat[i, j] = 0.0
                    self.table.setItem(i, j, item)
                item.setTextAlignment(Qt.AlignCenter)

        # 保存修改后的矩阵
        self.editor.set_matrix(mat)

        # 同步更新输入/输出向量表格
        self.update_vector_tables()
        self.update_output_vector()


    def init_default_vector(self):
        rows = self.table.rowCount()
        self.input_vector.setRowCount(rows)
        for i in range(rows):
            item = QTableWidgetItem("1")
            item.setTextAlignment(Qt.AlignCenter)
            self.input_vector.setItem(i, 0, item)

    def update_vector_tables(self):
        mat = self.editor.get_matrix()
        rows, cols = mat.shape
        # 输入列向量
        self.input_vector.setRowCount(cols)
        self.input_vector.setColumnCount(1)
        self.input_vector.horizontalHeader().hide()
        self.input_vector.verticalHeader().hide()

        for i in range(cols):
            item = self.input_vector.item(i, 0)
            if not item or item.text().strip() == "":
                item = QTableWidgetItem("1")  # 默认填 1
                self.input_vector.setItem(i, 0, item)
            item.setTextAlignment(Qt.AlignCenter)

        cell_size = 65
        for i in range(cols):
            self.input_vector.setRowHeight(i, cell_size)
        self.input_vector.setColumnWidth(0, cell_size)
        # 输出列向量
        self.output_vector.setRowCount(rows)
        self.output_vector.setColumnCount(1)
        self.output_vector.horizontalHeader().hide()
        self.output_vector.verticalHeader().hide()
        self.output_vector.setEditTriggers(QTableWidget.NoEditTriggers) # 只读

        for i in range(rows):
            item = self.output_vector.item(i, 0)
            if not item:
                item = QTableWidgetItem("")
                self.output_vector.setItem(i, 0, item)
            item.setTextAlignment(Qt.AlignCenter)

        for i in range(rows):
            self.output_vector.setRowHeight(i, cell_size)
        self.output_vector.setColumnWidth(0, cell_size)

    def get_input_vector(self):
        rows = self.input_vector.rowCount()
        vec = np.zeros((rows, 1), dtype=float)
        for i in range(rows):
            item = self.input_vector.item(i, 0)
            text = item.text().strip() if item else ""
            if text == "" or text == " ":
                vec[i, 0] = 0.0
            else:
                try:
                    num = float(text)
                    vec[i, 0] = num
                except:
                    vec[i, 0] = 0.0
        return vec

    def set_output_vector(self, vec):
        rows = vec.shape[0]
        self.output_vector.setRowCount(rows)

        for i in range(rows):
            v = vec[i, 0]
            if np.isnan(v):
                text = " "
            elif float(v).is_integer():
                text = str(int(v))
            else:
                text = str(v)

            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            self.output_vector.setItem(i, 0, item)

    def update_output_vector(self):
        mat = self.editor.get_matrix()
        rows, cols = self.table.rowCount(), self.table.columnCount()
        for i in range(rows):
            for j in range(cols):
                item = self.table.item(i, j)
                text = item.text().strip() if item else ""
                if text in ["", " "]:
                    mat[i, j] = 0.0  # 空格当0
                else:
                    try:
                        mat[i, j] = float(text)
                    except:
                        mat[i, j] = 0.0
        self.editor.set_matrix(mat)
        try:
            vec = self.get_input_vector()
            result = self.editor.multiply_vector(vec)
            self.set_output_vector(result)
        except Exception as e:
            print("Matrix multiply error:", e)

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
            self.plot_3d_bar()
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
        self.update_output_vector()
        self.plot_3d_bar()

    def random_fill_matrix(self): # 填满随机-9~9整数
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        rand_mat = np.random.randint(-9, 10, size=(rows, cols)).astype(float)
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
        self.update_output_vector()
        self.plot_3d_bar()

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
            sys_numbered_str = f"[{self.sys_operation_count}] Apply row operation"
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

    def perform_rref(self): # 计算rref并显示
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
            self.rref_window = MatrixDisplayWindow(rref_np, title="RREF", event="rref")
            self.rref_window.show()
        except Exception as e:
            QMessageBox.warning(self, "RREF Error", str(e))

    def compute_nullspace(self): # 计算kernel并显示
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
            if nullspace:
                # 转置
                vectors = np.array([v.T.tolist()[0] for v in nullspace], dtype=float).T.tolist()
            else:
                vectors = []
            self.null_window = MatrixDisplayWindow(vectors, title="Kernel", show_zero_if_empty=True, event="kernel")
            self.null_window.show()
        except Exception as e:
            QMessageBox.warning(self, "Null Space Error", str(e))

    def plot_3d_bar(self): # 3D绘图
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

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        ax.set_facecolor("#000000")
        self.figure.patch.set_facecolor("#000000")
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.ravel()
        y = y.ravel()
        z = np.zeros_like(x)
        dz = mat.ravel()
        dx = 0.3 * (cols / max(rows, cols))
        dy = 0.3 * (rows / max(rows, cols))
        ax.bar3d(x, y, z, dx, dy, dz, color="#ffffff", edgecolor="#000000", alpha=0.6)
        ax.set_xlabel('x: row', color='#ffffff')
        ax.set_ylabel('y: column', color='#ffffff')
        ax.set_zlabel('z: value', color='#ffffff')
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols + 1), color='#ffffff')
        ax.set_yticklabels(np.arange(1, rows + 1), color='#ffffff')
        ax.tick_params(axis='z', colors='#ffffff')
        for xi, yi, zi in zip(x, y, dz):
            label = f'{int(zi)}' if zi.is_integer() else f'{zi:.1f}'
            ax.text(xi + dx / 2, yi + dy / 2, zi + 0.2, label,
                    color="#FFFF00", ha='center', va='bottom', fontsize=10)
        ax.set_title('◄ Your Matrix ►', color='#ffffff', fontsize=12)
        ax.view_init(elev=30, azim=-120)
        gray = (0.2, 0.2, 0.2, 0)
        ax.xaxis.set_pane_color(gray)
        ax.yaxis.set_pane_color(gray)
        ax.zaxis.set_pane_color(gray)
        ax.xaxis.line.set_color("#FF0000")
        ax.yaxis.line.set_color("#0000FF")
        ax.zaxis.line.set_color("#00FF00")
        ax.grid(False)
        grid_x = np.arange(-0.3, cols + 0.7, 1)
        grid_y = np.arange(-0.3, rows + 0.7, 1)
        for gx in grid_x:
            ax.plot([gx, gx], [-0.3, rows - 0.3], zs=0, zdir='z', color="#ffffff", linewidth=0.5)
        for gy in grid_y:
            ax.plot([-0.3, cols - 0.3], [gy, gy], zs=0, zdir='z', color='#ffffff', linewidth=0.5)
        self.canvas.setStyleSheet("border: 2px solid white;")

        self.canvas.draw()

    def compute_eigenvectors(self):
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except:
                            pass
            sym_mat = Matrix(mat.tolist())
            eigen_data = sym_mat.eigenvects()  # 返回特征值/重数/向量列表
            self.eigen_window = EigenDisplayWindow(eigen_data)
            self.eigen_window.show()

        except:
            self.error_window = MessageWindow()
            self.error_window.show()

    def compute_diagonalize(self):
        try:
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            mat = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        try:
                            mat[i, j] = float(item.text())
                        except:
                            pass
            sym_mat = Matrix(mat.tolist())
            P, D = sym_mat.diagonalize()
            P_np = np.array(P.tolist(), dtype=float)
            D_np = np.array(D.tolist(), dtype=float)
            self.diag_window = DiagonalizeDisplayWindow(P_np, D_np)
            self.diag_window.show()

        except:
            self.error_window = MessageWindow()
            self.error_window.show()
