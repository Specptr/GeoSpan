STYLESHEET = """
            /* 整个窗口的背景色和默认字体颜色 */
            QWidget {
                background-color: #1C1C1C;   /* 深灰背景 */
                color: #ffffff;              /* 白色文字*/
            }

            /* 所有按钮的样式 */
            QPushButton {
                background-color: #3B3B3B;   /* 深灰按钮背景 */
                color: #FFFFFF;              /* 白色文字 */
                border-radius: 10px;         /* 圆角半径 */
                border: 3px solid #0F0F0F;   /* 深色边框 */
                padding: 6px 12px;           /* 内边距 */
            }

            /* 所有输入框（QLineEdit）的样式 */
            QLineEdit {
                background-color: #181818;   /* 更深的背景色 */
                color: #ffffff;              /* 白色文字 */
                border: 1px solid #999999;   /* 灰色边框 */
                border-radius: 10px;         /* 圆角 */
                padding: 4px 8px;            /* 内边距 */
                font-size: 20px;             /* 字体大小 */
                font-family: Consolas;       /* 字体类型 */
            }

            /* 表格整体样式 */
            QTableWidget {
                background-color: #1C1C1C;   /* 表格背景色 */
                color: #FFFFFF;              /* 表格文字颜色 */
                gridline-color: #FFFFFF;     /* 网格线颜色 */
            }

            /* 表格单元格样式 */
            QTableWidget::item {
                background-color: #2A2A2A;   /* 单元格背景色 */
                border: 2px solid #666666;   /* 单元格边框粗细和颜色 */
                padding: 4px;                /* 单元格内边距 */

            QTextEdit {
                border: 10px solid #999999;
                border-radius: 10px;
                padding: 6px;
                background-color: #f9f9f9;
                font-family: Consolas;
                font-size: 12px;
            }


            }
        """
