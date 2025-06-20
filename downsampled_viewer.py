#!/usr/bin/env python3
"""
Memory-efficient grid visualization for ndv.
Assembles downsampled image tiles into a seamless mosaic view with MIP across channels.
"""

import re
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import tifffile as tf
from PIL import Image
from PyQt6.QtCore import Qt, QPoint, QByteArray, QBuffer, QIODevice
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QAction

# Pattern for acquisitions: {region}_{fov}_{z_layer}_{imaging_modality}_{channel_info}_{suffix}.tiff
# Examples: C5_0_0_Fluorescence_405_nm_Ex.tiff, D6_2_3_Brightfield_BF_Ex.tiff
FPATTERN = re.compile(
    r"(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<modality>[^_]+)_(?P<channel>[^_]+)_.*\.tiff?", re.IGNORECASE
)

class MosaicWidget(QLabel):
    """Widget displaying seamless mosaic of tiles with coordinate tracking."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.coordinates = {}  # {fov: (x_mm, y_mm)}
        self.fov_grid = {}  # {(row, col): fov}
        self.tile_size = 75  # Reduced from 150 for 4x faster processing
        self.grid_dims = (0, 0)  # (n_rows, n_cols)
        
        # Coordinate mapping parameters
        self.mm_per_pixel_x = 0.0
        self.mm_per_pixel_y = 0.0
        self.origin_mm = (0.0, 0.0)
        self._mapping_initialized = False
        
    def set_grid_data(self, coordinates: Dict[int, Tuple[float, float]], 
                      fov_grid: Dict[Tuple[int, int], int],
                      grid_dims: Tuple[int, int]):
        """Set grid configuration data."""
        self.coordinates = coordinates
        self.fov_grid = fov_grid
        self.grid_dims = grid_dims
        
        # Calculate pixel to mm mapping
        self._calculate_pixel_mapping()
        
    def _calculate_pixel_mapping(self):
        """Pre-calculate pixel position to mm coordinate mapping."""
        if not self.coordinates:
            self._mapping_initialized = False
            return
            
        x_coords = [c[0] for c in self.coordinates.values()]
        y_coords = [c[1] for c in self.coordinates.values()]
        
        if not x_coords or not y_coords:
            self._mapping_initialized = False
            return
            
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate mm per pixel
        n_rows, n_cols = self.grid_dims
        
        # Handle edge case where we have only one row or column
        if n_cols > 1:
            mm_per_tile_x = (x_max - x_min) / (n_cols - 1)
        else:
            mm_per_tile_x = 1.0  # Default value for single column
            
        if n_rows > 1:
            mm_per_tile_y = (y_max - y_min) / (n_rows - 1)
        else:
            mm_per_tile_y = 1.0  # Default value for single row
            
        self.mm_per_pixel_x = mm_per_tile_x / self.tile_size
        self.mm_per_pixel_y = mm_per_tile_y / self.tile_size
        self.origin_mm = (x_min, y_min)
        self._mapping_initialized = True
        
    def mouseMoveEvent(self, event):
        """Track mouse position and display coordinates."""
        if not self._mapping_initialized:
            return
            
        pos = event.position().toPoint()
        
        # Calculate mm coordinates from pixel position
        x_mm = self.origin_mm[0] + pos.x() * self.mm_per_pixel_x
        y_mm = self.origin_mm[1] + pos.y() * self.mm_per_pixel_y
        
        # Find which FOV we're over
        col = pos.x() // self.tile_size
        row = pos.y() // self.tile_size
        fov = self.fov_grid.get((row, col), -1)
        
        # Update status bar in parent
        parent = self.parent()
        while parent and not isinstance(parent, QMainWindow):
            parent = parent.parent()
            
        if parent and hasattr(parent, 'statusBar'):
            if fov >= 0:
                parent.statusBar().showMessage(
                    f"Position: ({x_mm:.3f}, {y_mm:.3f}) mm | "
                    f"Pixel: ({pos.x()}, {pos.y()}) | FOV: {fov}"
                )
            else:
                parent.statusBar().showMessage(
                    f"Position: ({x_mm:.3f}, {y_mm:.3f}) mm | "
                    f"Pixel: ({pos.x()}, {pos.y()})"
                )

class GridViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDV Grid Viewer - Fast Preview")
        self.resize(800, 600)  # Smaller default size for preview
        
        # Data storage
        self.acquisition_dir = None
        self.coordinates = {}  # {fov: (x_mm, y_mm)}
        self.channels = []
        self.file_map = {}  # {(channel, fov): filepath}
        self.cache_dir = None
        
        # Grid organization
        self.fov_grid = {}  # {(row, col): fov}
        self.grid_dims = (0, 0)  # (n_rows, n_cols)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("No acquisition loaded")
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        
        # Progress label for loading
        self.progress_label = QLabel("")
        info_layout.addWidget(self.progress_label)
        
        layout.addLayout(info_layout)
        
        # Mosaic area
        self.scroll = QScrollArea()
        self.mosaic_widget = MosaicWidget()
        self.scroll.setWidget(self.mosaic_widget)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        
        # Status bar for coordinates
        self.statusBar().showMessage("Ready - Open an acquisition to begin")
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Acquisition", self)
        open_action.triggered.connect(self._open_acquisition)
        file_menu.addAction(open_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Add quit action
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
    
    def _open_acquisition(self):
        """Open acquisition directory dialog."""
        print("[LOG] Opening acquisition directory dialog")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Acquisition Directory")
        
        if not dir_path:
            print("[LOG] No directory selected")
            return
        
        print(f"[LOG] Selected directory: {dir_path}")
        self.acquisition_dir = Path(dir_path)
        self._load_acquisition()
    
    def _load_acquisition(self):
        """Load acquisition data from directory."""
        print(f"[LOG] Loading acquisition from {self.acquisition_dir}")
        
        # Find timepoint directories
        timepoint_dirs = []
        for item in self.acquisition_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                timepoint_dirs.append(item)
        
        print(f"[LOG] Found {len(timepoint_dirs)} timepoint directories")
        
        if not timepoint_dirs:
            QMessageBox.warning(self, "Error", "No timepoint directories found")
            return
        
        # Use first timepoint
        timepoint_dir = sorted(timepoint_dirs, key=lambda x: int(x.name))[0]
        print(f"[LOG] Using timepoint directory: {timepoint_dir}")
        
        # Load coordinates
        coord_file = timepoint_dir / "coordinates.csv"
        if not coord_file.exists():
            QMessageBox.warning(self, "Error", f"coordinates.csv not found in {timepoint_dir}")
            return
        
        self._load_coordinates(coord_file)
        
        # Setup cache directory
        self.cache_dir = self.acquisition_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        print(f"[LOG] Cache directory: {self.cache_dir}")
        
        # Scan for TIFF files
        self._scan_files(timepoint_dir)
        
        # Update UI and build mosaic
        self._update_ui_after_load()
    
    def _load_coordinates(self, coord_file: Path):
        """Load FOV coordinates from CSV file."""
        print(f"[LOG] Loading coordinates from {coord_file}")
        self.coordinates.clear()
        
        try:
            with open(coord_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fov = int(row['fov'])
                    x_mm = float(row['x (mm)'])
                    y_mm = float(row['y (mm)'])
                    self.coordinates[fov] = (x_mm, y_mm)
            
            print(f"[LOG] Loaded {len(self.coordinates)} FOV coordinates")
            
        except Exception as e:
            print(f"[ERROR] Failed to load coordinates: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load coordinates: {e}")
    
    def _scan_files(self, timepoint_dir: Path):
        """Scan directory for TIFF files and organize by channel/FOV."""
        print(f"[LOG] Scanning {timepoint_dir} for TIFF files")
        
        self.file_map.clear()
        self.channels = set()
        
        # Find all TIFF files
        tiff_files = list(timepoint_dir.glob("*.tif")) + list(timepoint_dir.glob("*.tiff"))
        print(f"[LOG] Found {len(tiff_files)} TIFF files")
        
        for filepath in tiff_files:
            match = FPATTERN.search(filepath.name)
            if not match:
                print(f"[WARNING] File doesn't match pattern: {filepath.name}")
                continue
                
            region = match.group("region")
            fov = int(match.group("fov"))
            z_layer = int(match.group("z"))
            modality = match.group("modality")
            channel_info = match.group("channel")
            
            # Create a comprehensive channel identifier
            if modality.lower() == "fluorescence":
                # For fluorescence, use the wavelength/channel info
                channel = f"{channel_info}"
            else:
                # For other modalities (brightfield, etc.), use modality + channel
                channel = f"{modality}_{channel_info}"
            
            # Only include FOVs that have coordinates
            if fov not in self.coordinates:
                print(f"[WARNING] FOV {fov} not found in coordinates, skipping")
                continue
                
            self.channels.add(channel)
            key = (channel, fov)
            
            if key not in self.file_map:
                self.file_map[key] = []
            self.file_map[key].append(filepath)
        
        self.channels = sorted(list(self.channels))
        print(f"[LOG] Found channels: {self.channels}")
        print(f"[LOG] Mapped {len(self.file_map)} channel-FOV combinations")
    
    def _update_ui_after_load(self):
        """Update UI elements after loading acquisition."""
        if self.channels and self.coordinates:
            self.info_label.setText(
                f"Loaded: {len(self.coordinates)} FOVs, {len(self.channels)} channels | "
                f"Computing MIP across channels..."
            )
            
            # Build grid structure
            self._build_grid()
            
            # Create MIP mosaic
            self._create_mip_mosaic()
        else:
            self.info_label.setText("No valid data found")
    
    def _build_grid(self):
        """Build grid structure from coordinates."""
        print("[LOG] Building grid structure")
        
        if not self.coordinates:
            return
        
        # Get coordinate bounds and unique positions
        x_positions = sorted(set(c[0] for c in self.coordinates.values()))
        y_positions = sorted(set(c[1] for c in self.coordinates.values()))
        
        n_cols = len(x_positions)
        n_rows = len(y_positions)
        
        print(f"[LOG] Grid dimensions: {n_rows} rows x {n_cols} cols")
        
        # Create position to index mappings with tolerance
        x_to_col = {}
        y_to_row = {}
        
        tolerance = 0.001  # 1 micron tolerance
        
        for i, x in enumerate(x_positions):
            x_to_col[x] = i
            
        for i, y in enumerate(y_positions):
            y_to_row[y] = i
        
        # Build FOV grid
        self.fov_grid.clear()
        
        for fov, (x_mm, y_mm) in self.coordinates.items():
            # Find closest x position
            col = None
            for x_pos, idx in x_to_col.items():
                if abs(x_pos - x_mm) < tolerance:
                    col = idx
                    break
                    
            # Find closest y position
            row = None
            for y_pos, idx in y_to_row.items():
                if abs(y_pos - y_mm) < tolerance:
                    row = idx
                    break
                    
            if col is not None and row is not None:
                self.fov_grid[(row, col)] = fov
            else:
                print(f"[WARNING] Could not place FOV {fov} at ({x_mm}, {y_mm})")
        
        self.grid_dims = (n_rows, n_cols)
        
        # Update mosaic widget
        self.mosaic_widget.set_grid_data(self.coordinates, self.fov_grid, self.grid_dims)
        
        # Set widget size
        width = n_cols * self.mosaic_widget.tile_size
        height = n_rows * self.mosaic_widget.tile_size
        self.mosaic_widget.setFixedSize(width, height)
        
        print(f"[LOG] Grid built with {len(self.fov_grid)} tiles")
    
    def _get_middle_z_file(self, files: List[Path]) -> Path:
        """Select middle z-layer from list of files."""
        if len(files) == 1:
            return files[0]
        
        # Sort files by z-index
        z_files = []
        for f in files:
            match = FPATTERN.search(f.name)
            if match:
                z = int(match.group("z"))
                z_files.append((z, f))
        
        if not z_files:
            return files[0]  # Fallback
            
        z_files.sort(key=lambda x: x[0])
        mid_idx = len(z_files) // 2
        
        selected_file = z_files[mid_idx][1]
        print(f"[LOG] Selected z-layer {z_files[mid_idx][0]} from {len(z_files)} layers")
        
        return selected_file
    
    def _create_mip_mosaic(self):
        """Create MIP mosaic across all channels."""
        print("[LOG] Creating MIP mosaic across channels")
        
        n_rows, n_cols = self.grid_dims
        tile_size = self.mosaic_widget.tile_size
        
        # Create blank mosaic
        mosaic_img = Image.new('L', (n_cols * tile_size, n_rows * tile_size), 0)
        
        total_tiles = len(self.fov_grid)
        processed_tiles = 0
        
        # Process each FOV position
        for (row, col), fov in self.fov_grid.items():
            # Update progress
            processed_tiles += 1
            self.progress_label.setText(f"Processing tile {processed_tiles}/{total_tiles}")
            QApplication.processEvents()  # Update UI
            
            # Check if we have a cached MIP for this FOV
            mip_cache_path = self.cache_dir / f"mip_fov_{fov}_thumb.jpg"
            
            if mip_cache_path.exists():
                # Load cached MIP
                tile_img = Image.open(mip_cache_path)
                tile_img = tile_img.convert('L')
            else:
                # Generate MIP from all channels
                tile_img = self._generate_fov_mip(fov, mip_cache_path)
                
                if tile_img is None:
                    continue
            
            # Ensure correct size
            if tile_img.size != (tile_size, tile_size):
                tile_img = tile_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            
            # Paste into mosaic
            x = col * tile_size
            y = row * tile_size
            mosaic_img.paste(tile_img, (x, y))
        
        # Clear progress
        self.progress_label.setText("")
        
        # Update info label
        self.info_label.setText(
            f"Fast Preview: {len(self.coordinates)} FOVs, {len(self.channels)} channels"
        )
        
        # Display the mosaic
        self._display_mosaic(mosaic_img)
        
        print(f"[LOG] MIP mosaic created successfully")
    
    def _generate_fov_mip(self, fov: int, cache_path: Path) -> Optional[Image.Image]:
        """Generate fast composite image by selecting channel with highest mean intensity."""
        best_channel = None
        best_mean = -1
        best_image = None
        
        # Quick scan: find channel with highest mean intensity
        for channel in self.channels:
            key = (channel, fov)
            if key not in self.file_map:
                continue
                
            # Get middle z file
            file_path = self._get_middle_z_file(self.file_map[key])
            
            try:
                # Read image for mean calculation (small sample)
                img_array = tf.imread(file_path)
                
                # Quick downsample for mean calculation (every 10th pixel)
                downsampled = img_array[::10, ::10]
                mean_intensity = np.mean(downsampled)
                
                if mean_intensity > best_mean:
                    best_mean = mean_intensity
                    best_channel = channel
                    # Store the best image
                    best_image = img_array
                
            except Exception as e:
                print(f"[ERROR] Failed to load {channel} for FOV {fov}: {e}")
                continue
        
        if best_image is None:
            print(f"[WARNING] No channel data for FOV {fov}")
            return None
        
        try:
            # Fast conversion to 8-bit
            if best_image.dtype == np.uint16:
                # Simple bit shift for 16-bit images (much faster than percentiles)
                img_8bit = (best_image >> 8).astype(np.uint8)
            elif best_image.dtype != np.uint8:
                # Quick normalize for other types
                img_min, img_max = best_image.min(), best_image.max()
                if img_max > img_min:
                    img_8bit = ((best_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_8bit = np.zeros_like(best_image, dtype=np.uint8)
            else:
                img_8bit = best_image
            
            # Create PIL image
            pil_img = Image.fromarray(img_8bit)
            
            # Fast thumbnail with NEAREST for speed
            pil_img.thumbnail((self.mosaic_widget.tile_size, self.mosaic_widget.tile_size), 
                            Image.Resampling.NEAREST)
            
            # Save to cache with lower quality for speed
            pil_img.save(cache_path, "JPEG", quality=75)
            
            return pil_img
            
        except Exception as e:
            print(f"[ERROR] Failed to generate composite for FOV {fov}: {e}")
            return None
    
    def _display_mosaic(self, mosaic_img: Image.Image):
        """Convert PIL Image to QPixmap and display."""
        try:
            # Convert to RGB for display
            if mosaic_img.mode != 'RGB':
                mosaic_img = mosaic_img.convert('RGB')
            
            # Convert PIL Image to QPixmap using QImage
            img_data = mosaic_img.tobytes('raw', 'RGB')
            
            # Create QImage
            qimage = QImage(img_data, 
                          mosaic_img.width, 
                          mosaic_img.height, 
                          mosaic_img.width * 3,  # bytes per line
                          QImage.Format.Format_RGB888)
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            # Display
            self.mosaic_widget.setPixmap(pixmap)
            
            print(f"[LOG] Displayed MIP mosaic: {mosaic_img.width}x{mosaic_img.height} pixels")
            
        except Exception as e:
            print(f"[ERROR] Failed to display mosaic: {e}")
            QMessageBox.critical(self, "Display Error", f"Failed to display mosaic: {e}")

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("NDV Grid Viewer - MIP")
    app.setOrganizationName("NDV")
    
    # Create and show viewer
    viewer = GridViewer()
    viewer.show()
    
    # Load acquisition if provided as argument
    if len(sys.argv) > 1:
        acquisition_path = Path(sys.argv[1])
        if acquisition_path.exists() and acquisition_path.is_dir():
            viewer.acquisition_dir = acquisition_path
            viewer._load_acquisition()
        else:
            print(f"[WARNING] Invalid path provided: {sys.argv[1]}")
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()