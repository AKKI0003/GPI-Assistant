"""
Desktop mascot for the GPI Knowledge Assistant.

A small always-on-top, borderless, animated sprite that floats in the
bottom-right corner of the screen, like the old "Clippy" assistant.

What it does (and only this):
  - Plays an idle animation loop, forever, in the corner of the screen.
  - Left-click  -> opens ASSISTANT_URL (see below) in the default web
                   browser.
  - Right-click -> a small menu: Open Assistant, toggle "Run on Windows
                   startup", Quit.
  - Left-click-drag -> reposition it anywhere on screen for this session.

It never sends or receives chat messages itself -- it is purely a UI
launcher, nothing more.

Requirements: Python's built-in `tkinter` (ships with almost every
standard Windows Python install) and `Pillow` (`pip install --user
pillow` if you don't already have it -- the --user flag installs it
into your own user site-packages, no administrator rights needed).

No administrator access is used or required anywhere in this file.
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox
from pathlib import Path

from PIL import Image, ImageTk

# ─── Settings you're most likely to want to change ─────────────────────────
#
# The one address the mascot opens on left-click. Change ONLY this
# string to point the mascot at a different deployment -- nothing else
# in this file needs to change. Must be a full URL including the
# scheme (https:// / http://).
ASSISTANT_URL = "https://mounted-enviably-tapeless.ngrok-free.dev/app"

# ─── Paths ──────────────────────────────────────────────────────────────────

MASCOT_DIR   = Path(__file__).resolve().parent
ASSETS_DIR   = MASCOT_DIR / "assets"
SETTINGS_FILE = MASCOT_DIR / "mascot_settings.json"

# The project root is assumed to be the parent of this mascot/ folder.
# Kept only for the Windows-startup .vbs launcher below, which needs a
# path to re-launch mascot.py itself -- it is no longer used to locate
# a local web/app.py, since the mascot now opens ASSISTANT_URL instead
# of spawning a local server.
PROJECT_ROOT = MASCOT_DIR.parent

TRANSPARENT_KEY = "#ff00ff"   # must match the magenta used when the
                               # sprite frames were composited (see
                               # assets/frame_NN.png)
FRAME_DELAY_MS  = 2000         # idle animation speed
MARGIN_PX       = 24           # gap from the screen's bottom-right corner

STARTUP_FOLDER = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
STARTUP_VBS    = STARTUP_FOLDER / "GPIAssistantMascot.vbs"


# ─── Settings (just remembers whether startup launch is enabled) ──────────

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {"run_on_startup": False}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    except Exception:
        pass  # non-fatal -- worst case the toggle just doesn't persist


# ─── Windows per-user Startup folder helpers (NO admin rights needed) ─────
#
# %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup is a per-user
# folder every normal Windows account can write to -- this is exactly
# the folder Windows itself uses for "run this when I log in" shortcuts,
# and dropping a script into it is a completely ordinary file-write, not
# a system-level change. Anything placed here (.lnk, .vbs, .bat, .exe)
# runs automatically the next time this Windows user logs in.
#
# A .vbs launcher is used (rather than a .bat file) so that logging in
# doesn't flash a visible console window -- it silently starts
# `pythonw.exe` (the windowless variant of Python that ships alongside
# every normal python.exe) running this same mascot.py.

def _pythonw_path() -> str:
    """Prefer pythonw.exe (no console window) next to the current
    Python interpreter; fall back to the interpreter actually running
    this script if pythonw.exe isn't found (e.g. some virtualenvs)."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def enable_startup() -> bool:
    """Create the per-user Startup entry. Returns True on success."""
    try:
        STARTUP_FOLDER.mkdir(parents=True, exist_ok=True)
        pythonw = _pythonw_path()
        script  = str(MASCOT_DIR / "mascot.py")
        vbs_content = (
            'Set WshShell = CreateObject("WScript.Shell")\n'
            f'WshShell.Run """{pythonw}"" ""{script}""", 0, False\n'
        )
        STARTUP_VBS.write_text(vbs_content)
        return True
    except Exception as exc:
        messagebox.showerror("Mascot", f"Couldn't enable startup launch:\n{exc}")
        return False


def disable_startup() -> bool:
    """Remove the per-user Startup entry, if present. Returns True on
    success (including if it was already absent)."""
    try:
        if STARTUP_VBS.exists():
            STARTUP_VBS.unlink()
        return True
    except Exception as exc:
        messagebox.showerror("Mascot", f"Couldn't disable startup launch:\n{exc}")
        return False


# ─── The mascot window itself ──────────────────────────────────────────────

class Mascot:
    def __init__(self) -> None:
        self.settings = load_settings()
        # Keep the actual Startup-folder state and the settings file in
        # sync in case someone deleted the .vbs by hand, etc.
        self.settings["run_on_startup"] = STARTUP_VBS.exists()
        save_settings(self.settings)

        self.root = tk.Tk()
        self.root.overrideredirect(True)          # no title bar / borders
        self.root.wm_attributes("-topmost", True)  # always on top
        self.root.configure(bg=TRANSPARENT_KEY)
        # Windows-only: make the magenta key color invisible, so only
        # the sprite's own silhouette shows on the desktop.
        try:
            self.root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass  # non-Windows: window will just show a magenta square

        self.frames = self._load_frames()
        self.frame_index = 0

        self.label = tk.Label(self.root, bd=0, bg=TRANSPARENT_KEY)
        self.label.pack()

        self._position_bottom_right()
        self._bind_events()
        self._animate()

        # A separate lock so a double-click can't open two browser tabs
        # at once.
        self._launch_in_progress = False

    # ---- assets -------------------------------------------------------

    def _load_frames(self):
        paths = sorted(ASSETS_DIR.glob("frame_*.png"))
        if not paths:
            raise FileNotFoundError(
                f"No sprite frames found in {ASSETS_DIR} -- expected "
                f"frame_00.png, frame_01.png, ..."
            )
        return [ImageTk.PhotoImage(Image.open(p)) for p in paths]

    # ---- window placement ---------------------------------------------

    def _position_bottom_right(self) -> None:
        self.root.update_idletasks()
        w = self.frames[0].width()
        h = self.frames[0].height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = sw - w - MARGIN_PX
        y = sh - h - MARGIN_PX
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---- animation ------------------------------------------------------

    def _animate(self) -> None:
        self.label.configure(image=self.frames[self.frame_index])
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.root.after(FRAME_DELAY_MS, self._animate)

    # ---- interaction ------------------------------------------------------

    def _bind_events(self) -> None:
        self.label.bind("<Button-1>", self._on_left_press)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.bind("<ButtonRelease-1>", self._on_left_release)
        self.label.bind("<Button-3>", self._on_right_click)

        self._drag_start = None
        self._dragged = False

    def _on_left_press(self, event) -> None:
        self._drag_start = (event.x_root, event.y_root,
                             self.root.winfo_x(), self.root.winfo_y())
        self._dragged = False

    def _on_drag(self, event) -> None:
        if self._drag_start is None:
            return
        start_x, start_y, win_x, win_y = self._drag_start
        dx, dy = event.x_root - start_x, event.y_root - start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self._dragged = True
        self.root.geometry(f"+{win_x + dx}+{win_y + dy}")

    def _on_left_release(self, event) -> None:
        # Only treat this as a "click" (launch the assistant) if the
        # mouse didn't actually move -- otherwise every drag would also
        # open a browser tab, which isn't what anyone wants.
        if not self._dragged:
            self.launch_assistant()
        self._drag_start = None

    def _on_right_click(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Open Assistant", command=self.launch_assistant)
        menu.add_separator()
        startup_var = tk.BooleanVar(value=self.settings.get("run_on_startup", False))
        menu.add_checkbutton(
            label="Run mascot on Windows startup",
            variable=startup_var,
            command=lambda: self._toggle_startup(startup_var.get()),
        )
        menu.add_separator()
        menu.add_command(label="Quit mascot", command=self.root.destroy)
        menu.tk_popup(event.x_root, event.y_root)

    def _toggle_startup(self, enabled: bool) -> None:
        ok = enable_startup() if enabled else disable_startup()
        if ok:
            self.settings["run_on_startup"] = enabled
            save_settings(self.settings)

    # ---- launching the actual assistant --------------------------------

    def launch_assistant(self) -> None:
        """Opens ASSISTANT_URL (set at the top of this file) in the
        system's default web browser. The mascot never talks to the
        assistant directly -- this is its entire job."""
        if self._launch_in_progress:
            return
        self._launch_in_progress = True
        try:
            opened = webbrowser.open(ASSISTANT_URL, new=2)
            if not opened:
                messagebox.showerror(
                    "Mascot",
                    f"Couldn't open a browser for:\n{ASSISTANT_URL}",
                )
        except Exception as exc:
            messagebox.showerror("Mascot", f"Couldn't open the assistant:\n{exc}")
        finally:
            # Small delay before allowing another launch, so an
            # accidental double-click doesn't spawn two browser tabs.
            self.root.after(3000, lambda: setattr(self, "_launch_in_progress", False))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Mascot().run()
