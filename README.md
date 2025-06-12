# NDV Grid Viewer

Memory-efficient grid visualization for ndv microscopy acquisitions.

## Install
```bash
pip install -r requirements.txt
python create_desktop_shortcut.py
```

## Usage
```bash
python downsampled_viewer.py [acquisition_dir]
```
Or use desktop shortcut.

## Expected Structure
```
acquisition_dir/
├── 0/                    # timepoint
│   ├── coordinates.csv   # fov,x (mm),y (mm)
│   └── *.tiff           # manual_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
└── cache/               # auto-generated
```

## Controls
- **Left-click**: Select tiles
- **Right-click**: View full/MIP/stitch
- **Dropdown**: Switch channels
