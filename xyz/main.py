import sys
from PyQt6.QtWidgets import QApplication
from ui.home_window import TrafficVisionApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficVisionApp()
    window.show()
    sys.exit(app.exec())