#!/usr/bin/env python3
"""
Memory-efficient grid visualization for ndv.
Assembles downsampled image tiles into a mosaic view.
"""

import re
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tifffile as tf
from PIL import Image
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QScrollArea, QMenu, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QAction
import ndv

# Pattern for manual acquisitions: manual_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
FPATTERN = re.compile(
    r"manual_(?P<f>\d+)_(?P<z>\d+)_Fluorescence_(?P<wavelength>\d+)_nm_Ex\.tiff?", re.IGNORECASE
)

class TileWidget(QLabel):
    """Individual tile in the grid."""
    clicked = pyqtSignal(int, int)  # fov, mouse_button
    
    def __init__(self, fov: int, parent=None):
        super().__init__(parent)
        self.fov = fov
        self.selected = False
        self.setScaledContents(True)
        self.setStyleSheet("border: 2px solid transparent;")
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.fov, event.button())
    
    def set_selected(self, selected: bool):
        self.selected = selected
        color = "#00ff00" if selected else "transparent"
        self.setStyleSheet(f"border: 2px solid {color};")

class GridViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDV Grid Viewer")
        self.resize(1200, 800)
        
        # Data storage
        self.acquisition_dir = None
        self.coordinates = {}  # {fov: (x_mm, y_mm)}
        self.channels = []
        self.tiles = {}  # {fov: TileWidget}
        self.selected_fovs = set()
        self.file_map = {}  # {(channel, fov): filepath}
        self.cache_dir = None
        
        # Viewers for full resolution
        self._viewers = set()
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        controls.addWidget(self.channel_combo)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Grid area
        self.scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.scroll.setWidget(self.grid_widget)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Acquisition", self)
        open_action.triggered.connect(self._open_acquisition)
        file_menu.addAction(open_action)
    
    def _open_acquisition(self):
        print("[LOG] _open_acquisition called")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Acquisition Directory")
        print(f"[LOG] Selected directory: {dir_path}")
        if not dir_path:
            print("[LOG] No directory selected.")
            return
        
        self.acquisition_dir = Path(dir_path)
        self._load_acquisition()
    
    def _load_acquisition(self):
        print(f"[LOG] _load_acquisition called for {self.acquisition_dir}")
        # Find timepoint directories (e.g., "0", "1", etc.)
        timepoint_dirs = [d for d in self.acquisition_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        print(f"[LOG] Found timepoint directories: {[str(d) for d in timepoint_dirs]}")
        
        if not timepoint_dirs:
            print("[LOG] No timepoint directories found.")
            QMessageBox.warning(self, "Error", "No timepoint directories found")
            return
        
        # Use first timepoint for now
        timepoint_dir = sorted(timepoint_dirs)[0]
        print(f"[LOG] Using timepoint directory: {timepoint_dir}")
        
        # Load coordinates from timepoint directory
        coord_file = timepoint_dir / "coordinates.csv"
        print(f"[LOG] Looking for coordinates file: {coord_file}")
        if not coord_file.exists():
            print(f"[LOG] coordinates.csv not found in {timepoint_dir}")
            QMessageBox.warning(self, "Error", f"coordinates.csv not found in {timepoint_dir}")
            return
        
        self.coordinates.clear()
        with open(coord_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fov = int(row['fov'])
                x_mm = float(row['x (mm)'])
                y_mm = float(row['y (mm)'])
                self.coordinates[fov] = (x_mm, y_mm)
        print(f"[LOG] Loaded coordinates for {len(self.coordinates)} FOVs.")
        
        # Setup cache directory
        self.cache_dir = self.acquisition_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        print(f"[LOG] Cache directory set to: {self.cache_dir}")
        
        # Find all TIFF files in the timepoint directory
        self._scan_files(timepoint_dir)
        
        # Update UI
        self.channel_combo.clear()
        self.channel_combo.addItems(sorted(self.channels))
        print(f"[LOG] Channels found: {self.channels}")
        
        if self.channels:
            self._build_grid()
        else:
            print("[LOG] No channels found.")
    
    def _scan_files(self, timepoint_dir: Path):
        """Scan timepoint directory for TIFF files and organize by channel/fov."""
        print(f"[LOG] _scan_files called for {timepoint_dir}")
        self.file_map.clear()
        self.channels = set()
        
        # Get all TIFF files in this timepoint directory
        tiff_files = list(timepoint_dir.glob("*.tif*"))
        print(f"[LOG] Found {len(tiff_files)} TIFF files.")
        
        for filepath in tiff_files:
            if m := FPATTERN.search(filepath.name):
                fov = int(m.group("f"))
                wavelength = m.group("wavelength")
                channel = f"{wavelength}nm"  # Create a channel name from wavelength
                
                if fov in self.coordinates:
                    self.channels.add(channel)
                    # For multi-z, we'll select middle layer later
                    key = (channel, fov)
                    if key not in self.file_map:
                        self.file_map[key] = []
                    self.file_map[key].append(filepath)
        print(f"[LOG] Channels after scan: {self.channels}")
        print(f"[LOG] File map keys: {list(self.file_map.keys())}")
        self.channels = list(self.channels)
    
    def _get_middle_z_file(self, files: List[Path]) -> Path:
        """Select middle z-layer from list of files."""
        if len(files) == 1:
            return files[0]
        
        # Parse z indices
        z_files = []
        for f in files:
            if m := FPATTERN.search(f.name):
                z = int(m.group("z"))
                z_files.append((z, f))
        
        z_files.sort()
        mid_idx = len(z_files) // 2
        return z_files[mid_idx][1]
    
    def _build_grid(self):
        print("[LOG] _build_grid called")
        # Clear existing
        if self.grid_widget.layout():
            QWidget().setLayout(self.grid_widget.layout())
        
        self.tiles.clear()
        self.selected_fovs.clear()
        
        # Calculate grid bounds
        x_coords = [c[0] for c in self.coordinates.values()]
        y_coords = [c[1] for c in self.coordinates.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Estimate grid size (assuming regular spacing)
        n_cols = len(set(x_coords))
        n_rows = len(set(y_coords))
        print(f"[LOG] Grid size: {n_rows} rows x {n_cols} cols")
        
        # Create grid layout manually
        grid_layout = QVBoxLayout()
        self.grid_widget.setLayout(grid_layout)
        
        # Group FOVs by approximate row/column
        x_step = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 1
        y_step = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 1
        
        grid = {}  # {(row, col): fov}
        for fov, (x, y) in self.coordinates.items():
            col = round((x - x_min) / x_step) if x_step > 0 else 0
            row = round((y - y_min) / y_step) if y_step > 0 else 0
            grid[(row, col)] = fov
        
        # Build grid
        for row in range(n_rows):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setSpacing(2)
            
            for col in range(n_cols):
                if (row, col) in grid:
                    fov = grid[(row, col)]
                    tile = TileWidget(fov)
                    tile.clicked.connect(self._on_tile_clicked)
                    tile.setFixedSize(150, 150)  # Thumbnail size
                    self.tiles[fov] = tile
                    row_layout.addWidget(tile)
                else:
                    # Empty space
                    spacer = QLabel()
                    spacer.setFixedSize(150, 150)
                    row_layout.addWidget(spacer)
            
            row_layout.addStretch()
            grid_layout.addWidget(row_widget)
        
        grid_layout.addStretch()
        print(f"[LOG] Built grid with {len(self.tiles)} tiles.")
        
        # Load thumbnails for current channel
        if self.channel_combo.currentText():
            self._load_channel_thumbnails()
    
    def _load_channel_thumbnails(self):
        print("[LOG] _load_channel_thumbnails called")
        channel = self.channel_combo.currentText()
        print(f"[LOG] Current channel: {channel}")
        if not channel:
            print("[LOG] No channel selected.")
            return
        
        for fov, tile in self.tiles.items():
            cache_path = self.cache_dir / f"channel_{channel}_fov_{fov}_thumb.jpg"
            print(f"[LOG] Loading thumbnail for FOV {fov} at {cache_path}")
            
            if cache_path.exists():
                # Load from cache
                pixmap = QPixmap(str(cache_path))
                tile.setPixmap(pixmap)
            else:
                # Generate thumbnail
                key = (channel, fov)
                if key in self.file_map:
                    file_path = self._get_middle_z_file(self.file_map[key])
                    self._generate_thumbnail(file_path, cache_path, tile)
    
    def _generate_thumbnail(self, tiff_path: Path, cache_path: Path, tile: TileWidget):
        """Generate and cache thumbnail."""
        try:
            # Read TIFF
            img = tf.imread(tiff_path)
            
            # Convert to 8-bit for display
            if img.dtype != np.uint8:
                # Scale to 0-255
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            # Create PIL image and resize
            pil_img = Image.fromarray(img)
            pil_img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # Save to cache
            pil_img.save(cache_path, "JPEG", quality=85)
            
            # Display
            pixmap = QPixmap(str(cache_path))
            tile.setPixmap(pixmap)
            
        except Exception as e:
            print(f"Error generating thumbnail for {tiff_path}: {e}")
            tile.setText("Error")
    
    def _on_channel_changed(self, channel: str):
        """Handle channel selection change."""
        if channel and self.tiles:
            self._load_channel_thumbnails()
    
    def _on_tile_clicked(self, fov: int, button: int):
        """Handle tile click."""
        tile = self.tiles.get(fov)
        if not tile:
            return
        
        if button == Qt.MouseButton.LeftButton:
            # Toggle selection
            if fov in self.selected_fovs:
                self.selected_fovs.remove(fov)
                tile.set_selected(False)
            else:
                self.selected_fovs.add(fov)
                tile.set_selected(True)
        
        elif button == Qt.MouseButton.RightButton:
            # Context menu
            menu = QMenu(self)
            
            view_action = menu.addAction("View Full Resolution")
            view_action.triggered.connect(lambda: self._view_full_resolution(fov))
            
            if len(self.selected_fovs) > 1 and fov in self.selected_fovs:
                stitch_action = menu.addAction("Stitch Selected")
                stitch_action.triggered.connect(self._stitch_selected)
            
            mip_action = menu.addAction("View MIP")
            mip_action.triggered.connect(lambda: self._view_mip(fov))
            
            menu.exec(tile.mapToGlobal(QPoint(0, 0)))
    
    def _view_full_resolution(self, fov: int):
        """Open full resolution viewer for FOV."""
        channel = self.channel_combo.currentText()
        if not channel:
            return
        
        key = (channel, fov)
        if key not in self.file_map:
            return
        
        # Get all z-layers for this FOV
        files = sorted(self.file_map[key])
        
        try:
            if len(files) == 1:
                # Single file
                array = tf.imread(files[0])
            else:
                # Stack of files
                array = np.stack([tf.imread(f) for f in files])
            
            # Create viewer
            viewer = ndv.imshow(array, name=f"FOV {fov} - Channel {channel}")
            self._viewers.add(viewer)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load full resolution: {e}")
    
    def _view_mip(self, fov: int):
        """View maximum intensity projection."""
        channel = self.channel_combo.currentText()
        if not channel:
            return
        
        key = (channel, fov)
        if key not in self.file_map:
            return
        
        files = sorted(self.file_map[key])
        
        try:
            if len(files) == 1:
                # Single z-layer, just show it
                array = tf.imread(files[0])
            else:
                # Create MIP
                stack = np.stack([tf.imread(f) for f in files])
                array = np.max(stack, axis=0)
            
            viewer = ndv.imshow(array, name=f"MIP - FOV {fov} - Channel {channel}")
            self._viewers.add(viewer)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create MIP: {e}")
    
    def _stitch_selected(self):
        """Stitch selected tiles (placeholder for full implementation)."""
        if len(self.selected_fovs) < 2:
            return
        
        QMessageBox.information(self, "Stitch", 
            f"Would stitch {len(self.selected_fovs)} selected tiles.\n"
            "Full stitching implementation needed.")

def main():
    app = QApplication(sys.argv)
    viewer = GridViewer()
    viewer.show()
    
    # If directory provided as argument
    if len(sys.argv) > 1:
        viewer.acquisition_dir = Path(sys.argv[1])
        viewer._load_acquisition()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()