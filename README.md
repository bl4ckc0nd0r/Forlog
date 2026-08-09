# Forlog

<img width="1600" height="901" alt="Logo" src="https://github.com/user-attachments/assets/e3705fae-3226-465e-ac48-4ed24331577e" />

Forlog is a lightweight, no-frills voice recorder that transcribes your notes
to text — a running log of what you're doing while tinkering, coding, or
troubleshooting a problem.

I've always struggled with writing writeups, and this small tool is my
attempt to fix that.

Inspired by the recorder Dana Scully uses in *The X-Files* during her work
as a medical examiner, Forlog lets you build a timeline of your project to
use as raw material when writing it up afterward.

Drafting a first version becomes as simple as feeding the resulting `.txt`
to an AI with the right prompt, along with your screenshots (I may build a
way to integrate this directly into the app down the line).

By default, `forlog.py` stores your audio clips and transcripts in a
`Forlog Archive` folder inside `Documents`.

## Screenshots

<img width="1920" height="1080" alt="2026-08-09-163213_hyprshot" src="https://github.com/user-attachments/assets/c07ac9ee-25c9-423d-9339-86c1e78981f4" />
<img width="1920" height="1080" alt="2026-08-09-163112_hyprshot" src="https://github.com/user-attachments/assets/56860add-6ff1-4ae1-9de3-ba6e2000ebe0" />
<img width="1920" height="1080" alt="2026-08-09-162915_hyprshot" src="https://github.com/user-attachments/assets/f6fb43dd-0844-43bb-9519-e732d9063ce4" />
<img width="1920" height="1080" alt="2026-08-09-162859_hyprshot" src="https://github.com/user-attachments/assets/e0d52978-8176-44f1-8983-2fa47d04168e" />



## Installation

**1. Clone the repo:**
```bash
git clone git@github.com:bl4ckc0nd0r/Forlog.git
cd Forlog
```

**2. Install the dependencies:**
```bash
pip install textual faster-whisper --break-system-packages
```
Forlog also needs `pw-record` (PipeWire) available on your `PATH` for
recording — most Arch/Hyprland setups already have it, since it ships with
PipeWire.

**3. Make it executable:**
```bash
chmod +x forlog.py
```

**4. (Recommended) Create an alias** so you can launch it from anywhere:
```bash
echo 'alias forlog="python3 ~/Forlog/forlog.py"' >> ~/.bashrc
source ~/.bashrc
```
If you use zsh instead, swap `~/.bashrc` for `~/.zshrc`.

## Usage
```bash
forlog
```
- `r` — start/stop recording
- `d` — delete the selected clip (and its transcript line)
- `h` — go back to project picker
- `q` — quit
