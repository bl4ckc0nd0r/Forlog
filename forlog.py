#!/usr/bin/env python3
"""
forlog.py — retro TUI for voice logging while troubleshooting/homelabbing.

Flow: pick/create a project -> record short clips with 'r' -> each clip gets
transcribed in the background and appended to the project's transcript.txt
with a timestamp.

On-disk layout:
    ~/Documents/Forlog archive/projects/<project>/audio/<timestamp>.wav
    ~/Documents/Forlog archive/projects/<project>/transcript.txt
    ~/Documents/Forlog archive/projects/<project>/.index.json   (internal, maps
        audio filename -> exact transcript line, used only so 'delete clip'
        can remove the right line even if two clips share the same minute)

Install:
    pip install textual faster-whisper --break-system-packages

Requires pw-record (PipeWire) available on PATH.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import threading
import tqdm.std as _tqdm_std
_tqdm_std.tqdm._lock = threading.RLock()

import asyncio
import json
import signal
import wave
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label, Input, OptionList
from textual.worker import Worker, get_current_worker
from textual import work

BASE_DIR = Path.home() / "Documents" / "Forlog archive" / "projects"
MODEL_SIZE = "small"       # base/small/medium — go up if you have a GPU
DEVICE = "cpu"              # "cuda" if you have a supported GPU
COMPUTE_TYPE = "int8"       # float16 if using cuda

# Format used for entries appended to transcript.txt: [dd-mm-yy]-[hh:mm]
TRANSCRIPT_TIMESTAMP_FORMAT = "[%d-%m-%y]-[%H:%M]"
# Format used for audio filenames on disk (must stay filesystem-safe/sortable)
AUDIO_FILENAME_FORMAT = "%Y%m%d-%H%M%S"

_model = None  # loaded once, lazily


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def build_screenshot_theme() -> Theme:
    """Theme built from colors sampled pixel-by-pixel out of a real
    screenshot (dark indigo background + pink/orange accents)."""
    return Theme(
        name="screenshot",
        background="#180b44",   # dark indigo background
        surface="#1c0c45",      # panels, slightly lighter than the background
        panel="#270f46",        # nested boxes (e.g. 'locals'-style panels)
        foreground="#c577a3",   # general text, muted rose-mauve
        primary="#e784a5",      # bright pink (keywords, titles)
        secondary="#dd8b4d",    # orange/gold (numbers, emphasis)
        accent="#dd8b4d",       # panel borders
        warning="#dd8b4d",
        error="#db556d",        # coral-red (error borders)
        success="#dd8b4d",      # no green in the source screenshot; reuse gold
        dark=True,
    )


class ProjectScreen(Screen):
    """Project picker shown on startup and when going home."""

    CSS = """
    ProjectScreen {
        align: center middle;
    }
    #panel {
        width: 60;
        height: auto;
        border: heavy $accent;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        projects = sorted(p.name for p in BASE_DIR.iterdir() if p.is_dir())
        with VerticalScroll(id="panel"):
            yield Label("[b]>> FORLOG // select project[/b]")
            yield OptionList(*projects, id="project-list")
            yield Label("or type a new name and press enter:")
            yield Input(placeholder="new-project", id="new-project")

    def on_mount(self) -> None:
        # Focus the option list right away so the arrow keys navigate
        # between projects immediately, with no click needed first.
        self.query_one("#project-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        name = str(event.option.prompt)
        self.dismiss(name)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name:
            self.dismiss(name)


class VoiceLogApp(App):
    CSS = """
    Screen {
        background: $background;
        color: $foreground;
    }
    Header, Footer {
        background: $panel;
        color: $foreground;
    }
    #audio-list {
        width: 40%;
        border: heavy $accent;
        padding: 0 1;
    }
    #transcript-panel {
        width: 60%;
        border: heavy $accent;
        padding: 0 1;
    }
    ListView > ListItem {
        color: $foreground;
        background: $background;
    }
    ListView > ListItem.-highlight {
        background: $panel;
    }
    #status {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $success;
    }
    """

    BINDINGS = [
        ("r", "toggle_record", "Record/Stop"),
        ("d", "delete_audio", "Delete clip"),
        ("h", "home", "Home"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.project_dir: Path | None = None
        self.audio_dir: Path | None = None
        self.transcript_path: Path | None = None
        self.recording = False
        self._proc: asyncio.subprocess.Process | None = None
        self._current_wav: Path | None = None
        self._rec_start: datetime | None = None
        self._audio_paths: list[Path] = []
        # Guards concurrent writes to transcript.txt / .index.json, since
        # several clips can be transcribing in background threads at once.
        self._transcript_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ListView(id="audio-list")
            with VerticalScroll(id="transcript-panel"):
                yield Static("", id="transcript-view")
        yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.register_theme(build_screenshot_theme())
        self.theme = "screenshot"
        self.run_worker(self._select_project(), exclusive=True)

    def action_home(self) -> None:
        self.run_worker(self._select_project(), exclusive=True)

    async def _select_project(self) -> None:
        name = await self.push_screen_wait(ProjectScreen())
        self.project_dir = BASE_DIR / name
        self.audio_dir = self.project_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.project_dir / "transcript.txt"
        self.transcript_path.touch(exist_ok=True)
        self.title = f"FORLOG :: {name}"
        await self.refresh_audio_list()
        self.refresh_transcript_view()
        self.set_status(f"Project: {name} — press 'r' to record")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    async def refresh_audio_list(self) -> None:
        list_view = self.query_one("#audio-list", ListView)
        await list_view.clear()
        self._audio_paths = sorted(self.audio_dir.glob("*.wav"))
        for wav in self._audio_paths:
            dur = self._wav_duration(wav)
            label = f"{wav.stem}  ({dur:.1f}s)" if dur else f"{wav.stem}  (…)"
            await list_view.append(ListItem(Label(label)))

    async def action_delete_audio(self) -> None:
        list_view = self.query_one("#audio-list", ListView)
        index = list_view.index
        if index is None or not self._audio_paths:
            self.set_status("No clip selected to delete")
            return
        try:
            wav_path = self._audio_paths[index]
        except IndexError:
            return
        wav_path.unlink(missing_ok=True)
        removed_line = self._remove_transcript_entry(wav_path.stem)
        if removed_line:
            self.set_status(f"Deleted: {wav_path.stem} (and its transcript line)")
        else:
            self.set_status(f"Deleted: {wav_path.stem}")
        await self.refresh_audio_list()
        self.refresh_transcript_view()

    def refresh_transcript_view(self) -> None:
        text = self.transcript_path.read_text(encoding="utf-8") if self.transcript_path.exists() else ""
        self.query_one("#transcript-view", Static).update(text or "[dim](no transcriptions yet)[/dim]")
        self.query_one("#transcript-panel", VerticalScroll).scroll_end(animate=False)

    @staticmethod
    def _wav_duration(path: Path) -> float | None:
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return None

    def _index_path(self) -> Path:
        return self.project_dir / ".index.json"

    def _load_index(self) -> dict[str, str]:
        p = self._index_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_index(self, index: dict[str, str]) -> None:
        self._index_path().write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_transcript_entry(self, wav_stem: str) -> bool:
        """Remove the exact transcript.txt line that belongs to wav_stem, if
        one was ever written. Returns True if a line was actually removed."""
        with self._transcript_lock:
            index = self._load_index()
            line_to_remove = index.pop(wav_stem, None)
            if line_to_remove is None:
                # Clip was deleted before it finished transcribing, or was
                # never transcribed — nothing to strip from the document.
                return False
            if self.transcript_path.exists():
                lines = self.transcript_path.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    if line == line_to_remove:
                        del lines[i]
                        break
                self.transcript_path.write_text(
                    ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
                )
            self._save_index(index)
            return True

    async def action_toggle_record(self) -> None:
        if not self.recording:
            await self.start_recording()
        else:
            await self.stop_recording()

    async def start_recording(self) -> None:
        ts = datetime.now().strftime(AUDIO_FILENAME_FORMAT)
        self._current_wav = self.audio_dir / f"{ts}.wav"
        self._rec_start = datetime.now()
        self._proc = await asyncio.create_subprocess_exec(
            "pw-record", "--rate", "16000", "--channels", "1", str(self._current_wav)
        )
        self.recording = True
        self.set_status(f"● RECORDING {self._current_wav.stem} — press 'r' to stop")

    async def stop_recording(self) -> None:
        if self._proc:
            self._proc.send_signal(signal.SIGINT)
            await self._proc.wait()
        self.recording = False
        wav_path = self._current_wav
        self._current_wav = None
        self.set_status(f"Transcribing {wav_path.stem}...")
        await self.refresh_audio_list()
        self.transcribe_clip(wav_path)

    @work(thread=True, exclusive=False)
    def transcribe_clip(self, wav_path: Path) -> None:
        try:
            model = get_model()
            segments, _info = model.transcribe(str(wav_path), language="es", beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except FileNotFoundError:
            self.call_from_thread(
                self.set_status, f"Skipped {wav_path.stem}: deleted before it finished transcribing"
            )
            return
        except Exception as exc:
            self.call_from_thread(self.set_status, f"Transcription failed for {wav_path.stem}: {exc}")
            return

        # Recover the recording's actual datetime from the filename
        # (AUDIO_FILENAME_FORMAT) and render it in TRANSCRIPT_TIMESTAMP_FORMAT
        # for the transcript entry, e.g. [09-08-26]-[14:35].
        recorded_at = datetime.strptime(wav_path.stem, AUDIO_FILENAME_FORMAT)
        ts_label = recorded_at.strftime(TRANSCRIPT_TIMESTAMP_FORMAT)
        entry_line = f"{ts_label} {text}"

        with self._transcript_lock:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(entry_line + "\n")
            index = self._load_index()
            index[wav_path.stem] = entry_line
            self._save_index(index)

        self.call_from_thread(self._on_transcription_done, wav_path)

    def _on_transcription_done(self, wav_path: Path) -> None:
        self.set_status(f"Done: {wav_path.stem} — press 'r' to record another")
        self.refresh_transcript_view()
        self.call_next(self.refresh_audio_list)


if __name__ == "__main__":
    try:
        print("Loading Whisper model (this can take a while on first download)...")
        get_model()
    except Exception as exc:
        print(f"Could not load the Whisper model: {exc}")
        raise SystemExit(1)
    VoiceLogApp().run()
