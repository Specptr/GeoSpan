# GeoSpan by EnoLaice
# 11/6/2025 alpha
# 11/8/2025 v1.0
# 11/12/2025 v1.1
# 11/13/2025 v1.2
# 11/26/2025 v1.3

import sys
from GS_ui import QApplication, GeoSpanUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeoSpanUI()
    window.show()
    sys.exit(app.exec_())
