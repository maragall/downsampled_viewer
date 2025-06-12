#!/usr/bin/env python3
"""
create_desktop_shortcut.py – generate a Desktop launcher for the
NDV Grid Viewer on Ubuntu (XDG-compliant) and Windows.

Run from the project root:
    python3 create_desktop_shortcut.py

The script detects the current OS and writes a shortcut that executes
``python downsampled_viewer.py`` with the project icon found at
``ndv_grid_icon.png``.
"""

from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path

# ────────────────────────────── helpers ───────────────────────────────────

def find_icon(project_root: Path) -> Path:
    """Return the icon PNG in project root."""
    icon_path = project_root / "ndv_grid_icon.png"
    if icon_path.exists():
        return icon_path
    sys.exit("[ERROR] No icon PNG found (expected ndv_grid_icon.png)")

# Windows --------------------------------------------------------------

def windows_shortcut(project_root: Path, icon_png: Path, desktop: Path) -> None:
    """Create a .bat launcher and a .lnk shortcut on Windows."""
    bat_path = project_root / "start_ndv_grid_viewer.bat"
    bat_content = (
        "@echo off\r\n"
        f"cd /d \"{project_root}\"\r\n"
        "python downsampled_viewer.py\r\n"
    )
    bat_path.write_text(bat_content, encoding="utf-8")
    
    lnk_path = desktop / "NDV Grid Viewer.lnk"
    
    # Convert PNG to ICO for Windows (if PIL available)
    ico_path = None
    try:
        from PIL import Image
        ico_path = project_root / "ndv_grid_icon.ico"
        img = Image.open(icon_png)
        img.save(ico_path, format='ICO', sizes=[(256, 256)])
    except ImportError:
        print("[WARNING] PIL not available, using PNG icon")
    
    ps_cmd = (
        "$WshShell = New-Object -ComObject WScript.Shell; "
        f"$Shortcut = $WshShell.CreateShortcut('{lnk_path}'); "
        f"$Shortcut.TargetPath = '{bat_path}'; "
        f"$Shortcut.IconLocation = '{ico_path or icon_png}'; "
        f"$Shortcut.WorkingDirectory = '{project_root}'; "
        "$Shortcut.Save()"
    )
    
    subprocess.run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", ps_cmd,
    ], check=True)
    
    print(f"[OK] Windows shortcut created → {lnk_path}")

# Ubuntu / Linux -------------------------------------------------------

def ubuntu_shortcut(project_root: Path, icon_png: Path, desktop: Path) -> None:
    """Create an XDG .desktop launcher on Ubuntu/Linux."""
    desktop_file = desktop / "ndv_grid_viewer.desktop"
    python_exec = sys.executable  # absolute path to interpreter
    
    desktop_content = "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=NDV Grid Viewer",
        "Comment=Memory-efficient grid visualization for ndv",
        f"Path={project_root}",
        f"Exec={python_exec} {project_root}/downsampled_viewer.py",
        f"Icon={icon_png}",
        "Terminal=false",
        "Categories=Graphics;Science;Viewer;",
        ""
    ])
    
    desktop_file.write_text(desktop_content, encoding="utf-8")
    desktop_file.chmod(0o755)
    
    print(f"[OK] Ubuntu shortcut created → {desktop_file}")

# ─────────────────────────────── main ───────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parent
    icon_png = find_icon(project_root)
    
    desktop = Path.home() / "Desktop"
    desktop.mkdir(exist_ok=True)
    
    os_name = platform.system().lower()
    
    if os_name == "windows":
        windows_shortcut(project_root, icon_png, desktop)
    elif os_name == "linux":
        ubuntu_shortcut(project_root, icon_png, desktop)
    else:
        sys.exit("[ERROR] Unsupported OS: only Windows and Ubuntu are handled.")

if __name__ == "__main__":
    main()