from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from config import STYLESHEET

class MatrixDisplayWindow(QWidget):#对rref和kernel的窗口化显示
    def __init__(self, matrix, title="Result", show_zero_if_empty=False):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(575, 425)
        self.setStyleSheet(STYLESHEET)
        layout = QVBoxLayout()
        table = QTableWidget()#表格显示
        if show_zero_if_empty and not matrix:# 如果是空零空间，显示单个 0
            table.setRowCount(1)
            table.setColumnCount(1)
            item = QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, 0, item)
        else:
            table.setRowCount(len(matrix))
            table.setColumnCount(len(matrix[0]) if matrix else 0)
            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    # 格式化数值：整数不带小数，浮点保留最多 4 位有效数字
                    rounded = round(float(val), 4)
                    display = str(int(rounded)) if rounded.is_integer() else f"{rounded:.4f}".rstrip("0").rstrip(".")
                    item = QTableWidgetItem(display)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, j, item)
        table.horizontalHeader().hide() #隐藏表头
        table.verticalHeader().hide()
        cell_size = 80
        for i in range(table.rowCount()):
            table.setRowHeight(i, cell_size)
        for j in range(table.columnCount()):
            table.setColumnWidth(j, cell_size)
        layout.addWidget(table)
        self.setLayout(layout)