# Desktop Mascot

A small animated sprite that floats in the bottom-right corner of your
screen, always on top — like the old "Clippy" assistant. It's purely a
UI launcher:

- **Left-click** it → runs `streamlit run web/app.py` in the background
  (nothing else — it never sends it chat messages or talks to it directly).
- **Left-click + drag** → move it anywhere on screen for this session.
- **Right-click** → a small menu:
  - **Open Assistant** — same as left-click.
  - **Run mascot on Windows startup** — a checkbox. Toggle it on to have
    the mascot start automatically next time you log into Windows;
    toggle it off to stop that.
  - **Quit mascot** — closes it.

## Setup (no administrator access required, anywhere)

1. Put this whole `mascot/` folder inside your project, next to `web/`,
   so the folder layout looks like:
   ```
   your_project/
     web/
       app.py
     mascot/
       mascot.py
       assets/
       README.md   <- this file
   ```
   (If you'd rather keep it somewhere else, open `mascot.py` and edit
   the `PROJECT_ROOT` line near the top to point at the folder that
   contains your `web/app.py`.)

2. Make sure Pillow is installed for your Python. This is a normal,
   per-user package install — it does **not** need administrator rights:
   ```
   pip install --user pillow
   ```
   (`tkinter`, the other dependency, ships built into almost every
   standard Windows Python install already — nothing to do there.)

3. Run it:
   ```
   python mascot/mascot.py
   ```
   The sprite should appear in the bottom-right corner of your screen.

## About the "run on startup" feature

Turning on "Run mascot on Windows startup" writes one small script into:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
```

This is your own personal, per-user Windows folder — it's the exact
same mechanism Windows itself uses for "start this when I log in," and
writing to it is an ordinary file operation, not a system-level change.
**It does not require, and never asks for, administrator access.**
Turning the checkbox back off simply deletes that one file again.

If you ever want to check it by hand, you can open that folder directly
by typing `shell:startup` into the Windows Run dialog (Win+R).

## Troubleshooting

- **"Couldn't find the 'streamlit' command"** — Streamlit isn't
  installed for this Python, or isn't on your PATH. Try:
  `pip install --user streamlit`
- **Mascot doesn't look transparent / shows a magenta box** — this
  transparency trick (`-transparentcolor`) is Windows-specific. On
  other operating systems the mascot will still work, just with a
  solid magenta background instead of a cut-out sprite.
- **Mascot doesn't appear at login even with startup enabled** — check
  `shell:startup` for a file called `GPIAssistantMascot.vbs`. If it's
  missing, just toggle the checkbox off and on again from the mascot's
  right-click menu.
