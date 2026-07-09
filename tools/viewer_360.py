"""Minimal interactive 360° panorama viewer for APRS.
    python tools/viewer_360.py --root APRS_dataset --split test --index 0
    python tools/viewer_360.py --hf --split test --index 0
    python tools/viewer_360.py path/to/panorama.jpg
"""
import argparse
import math
import os
import sys

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget

FOV_H = 120.0
FOV_V = 90.0
LOOK_STEP = 5.0
SPHERE_R = 10.0
MARKER_R = 9.7


def look_dir(yaw_deg, pitch_deg, r=1.0):
    lo, la = math.radians(yaw_deg), math.radians(pitch_deg)
    cl = math.cos(la)
    return (r * cl * math.sin(lo), r * math.sin(la), -r * cl * math.cos(lo))


class GLViewer(QOpenGLWidget):
    viewChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, image_path, box=None, init_yaw=0.0, init_pitch=0.0,
                 parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.box = box
        self.yaw = init_yaw
        self.pitch = init_pitch
        self.texture_id = None
        self.last_pos = QtCore.QPoint()
        self.setFixedSize(1000, 560)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_view(self, yaw, pitch):
        self.yaw = ((yaw + 180) % 360) - 180
        self.pitch = max(-89.9, min(89.9, pitch))
        self.update()
        self.viewChanged.emit(self.yaw, self.pitch)

    def nudge(self, dyaw, dpitch):
        self.set_view(self.yaw + dyaw, self.pitch + dpitch)

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glClearColor(0.12, 0.14, 0.18, 1.0)
        self._load_texture(self.image_path)
        self._update_projection()

    def _load_texture(self, path):
        if path is None:
            print("No panorama path provided")
            return
        try:
            # Support file paths, PIL Image, and numpy arrays
            if isinstance(path, np.ndarray):
                # Convert BGR (OpenCV) to RGB, then to PIL
                if path.shape[2] == 3:
                    rgb = path[:, :, ::-1]  # BGR -> RGB
                    img = Image.fromarray(rgb).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    img = Image.fromarray(path).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(path, Image.Image):
                img = path.convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(path, str) and os.path.exists(path):
                img = Image.open(path).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
            else:
                print(f"Panorama not found: {path}")
                return

            ix, iy = img.size
            data = img.tobytes("raw", "RGBA", 0, 1)
            self.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, ix, iy, GL_RGBA,
                              GL_UNSIGNED_BYTE, data)
        except Exception as e:
            print(f"Texture load failed: {e}")

    def _update_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        near, far = 0.1, 100.0
        top = near * math.tan(math.radians(FOV_V / 2.0))
        right = near * math.tan(math.radians(FOV_H / 2.0))
        glFrustum(-right, right, -top, top, near, far)
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, w, h):
        ratio = self.devicePixelRatio()
        glViewport(0, 0, int(w * ratio), int(h * ratio))
        self._update_projection()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        fx, fy, fz = look_dir(self.yaw, self.pitch)
        gluLookAt(0, 0, 0, fx, fy, fz, 0, 1, 0)
        self._draw_sphere()
        self._draw_box()

    def _draw_sphere(self, nlon=128, nlat=64):
        if not self.texture_id:
            return
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1, 1, 1, 1)
        for i in range(nlat):
            lat0 = 90 - 180 * i / nlat
            lat1 = 90 - 180 * (i + 1) / nlat
            glBegin(GL_QUAD_STRIP)
            for j in range(nlon + 1):
                lon = -180 + 360 * j / nlon
                u = (180 - lon) / 360
                for lat in (lat0, lat1):
                    v = (90 - lat) / 180
                    x, y, z = look_dir(lon, lat, SPHERE_R)
                    glTexCoord2f(u, v)
                    glVertex3f(x, y, z)
            glEnd()

    def _draw_box(self):
        if not self.box:
            return
        pts = self._box_loop(*self.box)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(7.0)
        glColor3f(1, 1, 1)
        self._line_loop(pts)
        glLineWidth(3.0)
        glColor3f(0.0, 1.0, 0.0)
        self._line_loop(pts)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

    @staticmethod
    def _box_loop(y0, y1, p0, p1, n=24):
        pts = []
        pts += [look_dir(y0 + (y1 - y0) * i / n, p1, MARKER_R) for i in range(n + 1)]
        pts += [look_dir(y1, p1 + (p0 - p1) * i / n, MARKER_R) for i in range(n + 1)]
        pts += [look_dir(y1 + (y0 - y1) * i / n, p0, MARKER_R) for i in range(n + 1)]
        pts += [look_dir(y0, p0 + (p1 - p0) * i / n, MARKER_R) for i in range(n + 1)]
        return pts

    @staticmethod
    def _line_loop(pts):
        glBegin(GL_LINE_LOOP)
        for x, y, z in pts:
            glVertex3f(x, y, z)
        glEnd()

    def mousePressEvent(self, e):
        self.last_pos = e.pos()

    def mouseMoveEvent(self, e):
        dx, dy = e.x() - self.last_pos.x(), e.y() - self.last_pos.y()
        self.last_pos = e.pos()
        self.set_view(self.yaw - dx * 0.2, self.pitch - dy * 0.2)

    def keyPressEvent(self, e):
        k = e.key()
        if k in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
            self.nudge(LOOK_STEP, 0)
        elif k in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
            self.nudge(-LOOK_STEP, 0)
        elif k in (QtCore.Qt.Key_W, QtCore.Qt.Key_Up):
            self.nudge(0, LOOK_STEP)
        elif k in (QtCore.Qt.Key_S, QtCore.Qt.Key_Down):
            self.nudge(0, -LOOK_STEP)
        else:
            super().keyPressEvent(e)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, image_path, box=None, init_yaw=0.0, init_pitch=0.0,
                 title=""):
        super().__init__()
        self.setWindowTitle("APRS 360° Viewer")
        self.setStyleSheet(
            "QMainWindow{background:#eef1f5;} QLabel{color:#1f2937;font-family:Arial;}")
        self._init_yaw, self._init_pitch = init_yaw, init_pitch

        self.viewer = GLViewer(image_path, box=box,
                               init_yaw=init_yaw, init_pitch=init_pitch)
        self.viewer.viewChanged.connect(self._on_view)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet(
            "font-family:Consolas,monospace;color:#2563eb;font-size:14px;font-weight:600;")
        hint = QtWidgets.QLabel(
            (title + "  ·  " if title else "") +
            "Drag / WASD / arrows to look · R reset · 🟩 target")
        hint.setStyleSheet("color:#6b7280;font-size:12px;")

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.status)
        bar.addStretch()
        bar.addWidget(hint)

        box_w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box_w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.addWidget(self.viewer, alignment=QtCore.Qt.AlignCenter)
        lay.addLayout(bar)
        self.setCentralWidget(box_w)

        self.viewer.set_view(init_yaw, init_pitch)
        self.viewer.setFocus()

    def _on_view(self, y, p):
        self.status.setText(f"θ: {y:+.1f}°   φ: {p:+.1f}°")

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_R:
            self.viewer.set_view(self._init_yaw, self._init_pitch)
        else:
            self.viewer.keyPressEvent(e)


def box_to_angular(box_norm):
    x, y, w, h = box_norm
    if w <= 0 or h <= 0:
        return None
    theta0 = 360.0 * x - 180.0
    theta1 = 360.0 * (x + w) - 180.0
    phi_top = 90.0 - 180.0 * y
    phi_bottom = 90.0 - 180.0 * (y + h)
    return (theta0, theta1, phi_bottom, phi_top)


def main():
    ap = argparse.ArgumentParser(description="Minimal APRS 360° panorama viewer")
    ap.add_argument("image", nargs="?", help="panorama image path (image-only mode)")
    ap.add_argument("--root",  default=None, help="APRS dataset root (loads a sample instead)")
    ap.add_argument("--hf", action="store_true", help="Load from HuggingFace dataset")
    ap.add_argument("--repo-id", default="FudanCVL/APRS_dataset", help="HuggingFace repository ID")
    ap.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--index", type=int, default=0, help="sample index")
    args = ap.parse_args()

    image_path, box, init_yaw, init_pitch, title = args.image, None, 0.0, 0.0, ""

    if args.hf:
        # Load from HuggingFace using APRSDataset.from_hub()
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from aprs import APRSDataset

        print(f"Loading from HuggingFace: {args.repo_id} (split={args.split}, index={args.index})...")
        dataset = APRSDataset.from_hub(repo_id=args.repo_id, split=args.split, token=args.token)

        if args.index >= len(dataset):
            print(f"Error: Index {args.index} out of range (dataset has {len(dataset)} samples)")
            sys.exit(1)

        sample = dataset[args.index]

        # Use preloaded image from HuggingFace (numpy array in BGR format)
        image_path = sample.load_image()  # Returns BGR numpy array
        box = box_to_angular(sample.box_norm)
        init_yaw, init_pitch = sample.init_theta, sample.init_phi
        title = f"[{sample.category}] {sample.instruction}"
        print(title)

    elif args.root:
        # Load from local dataset
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from aprs import APRSDataset
        sample = APRSDataset(args.root, split=args.split)[args.index]
        image_path = sample.image_path
        box = box_to_angular(sample.box_norm)
        init_yaw, init_pitch = sample.init_theta, sample.init_phi
        title = f"[{sample.category}] {sample.instruction}"
        print(title)
    elif not image_path:
        ap.error("provide an image path, --root for local dataset, or --hf for HuggingFace dataset")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(image_path, box=box, init_yaw=init_yaw,
                     init_pitch=init_pitch, title=title)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
