# GeoSpan by Specptr
# 11/6/2025 - 11/8/2025
# v1.0

import sys
from GS_ui import QApplication, GeoSpanUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeoSpanUI()
    window.show()
    sys.exit(app.exec_())
