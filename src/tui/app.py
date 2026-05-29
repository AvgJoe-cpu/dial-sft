"""
app.py — Textual TUI for dial-sft.

Provides:
  - Config builder functions (module-level): load_model_configs, load_stage_configs,
    load_base_config, build_override_string, build_multirun_overrides,
    resolve_config_preview
  - SetupScreen: two-panel config selection + override editor + live preview
  - MonitorScreen: job queue table + live log tail
  - DialSFTApp: screen router and entry point
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Select,
    Static,
)
from textual.worker import Worker

from src.tui.runner import Job, JobState, Runner

# ---------------------------------------------------------------------------
# Repo / conf paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_CONF_DIR = _REPO_ROOT / "conf"
_WEIGHTS_CHECKPOINTS = _REPO_ROOT / "weights" / "checkpoints"

# ---------------------------------------------------------------------------
# Config builder — module-level functions
# ---------------------------------------------------------------------------


def load_model_configs() -> dict[str, dict]:
    """Read all conf/model/*.yaml, return {name: parsed_dict}."""
    configs: dict[str, dict] = {}
    model_dir = _CONF_DIR / "model"
    if model_dir.exists():
        for yaml_file in sorted(model_dir.glob("*.yaml")):
            with yaml_file.open() as fh:
                data = yaml.safe_load(fh) or {}
            configs[yaml_file.stem] = data
    return configs


def load_stage_configs() -> dict[str, dict]:
    """Read all conf/stage/*.yaml, return {name: parsed_dict}."""
    configs: dict[str, dict] = {}
    stage_dir = _CONF_DIR / "stage"
    if stage_dir.exists():
        for yaml_file in sorted(stage_dir.glob("*.yaml")):
            with yaml_file.open() as fh:
                data = yaml.safe_load(fh) or {}
            configs[yaml_file.stem] = data
    return configs


def load_base_config() -> dict:
    """Read conf/config.yaml, return parsed dict."""
    config_path = _CONF_DIR / "config.yaml"
    if config_path.exists():
        with config_path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def build_override_string(stage: str, overrides: dict[str, str]) -> str:
    """Build the Hydra override string for a single run."""
    parts: list[str] = [f"stage={stage}"]
    for key, val in overrides.items():
        if val.strip():
            parts.append(f"{key}={val.strip()}")
    return " ".join(parts)


def build_multirun_overrides(
    stage: str, sweep_params: dict[str, list[str]]
) -> list[str]:
    """Return list of all override strings for the full cartesian product grid."""
    if not sweep_params:
        return [build_override_string(stage, {})]

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    override_strings: list[str] = []
    for combo in itertools.product(*value_lists):
        overrides = {keys[i]: combo[i] for i in range(len(keys))}
        override_strings.append(build_override_string(stage, overrides))
    return override_strings


def resolve_config_preview(stage: str, overrides: dict[str, str]) -> dict:
    """Merge base + stage + model + overrides into a single dict for preview."""
    base = load_base_config()
    stage_configs = load_stage_configs()
    model_configs = load_model_configs()

    # Remove Hydra defaults list (not useful for display)
    base.pop("defaults", None)

    # Merge stage config
    if stage in stage_configs:
        stage_cfg = dict(stage_configs[stage])
        stage_cfg.pop("defaults", None)
        base = _deep_merge(base, stage_cfg)

    # Merge model config (use the first available model if any)
    if model_configs:
        model_cfg = dict(next(iter(model_configs.values())))
        model_cfg.pop("defaults", None)
        base = _deep_merge(base, model_cfg)

    # Apply dot-notation overrides
    for key, val in overrides.items():
        if not val.strip():
            continue
        parts = key.split(".")
        node = base
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = val.strip()

    return base


# ---------------------------------------------------------------------------
# SetupScreen
# ---------------------------------------------------------------------------

_SCHEDULER_OPTIONS = [("linear", "linear"), ("cosine", "cosine")]
_LOSS_WEIGHT_OPTIONS = [("uniform", "uniform"), ("scheduler", "scheduler")]

_STAGE_CHAIN_STAGES = ["d0", "d1", "d2"]


class SetupScreen(Screen):
    """Configuration selection + override editing + live config preview."""

    TITLE = "Setup"

    def __init__(self, runner: Runner) -> None:
        super().__init__()
        self._runner = runner
        self._model_configs = load_model_configs()
        self._stage_configs = load_stage_configs()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="setup-body"):
            with Vertical(id="left-panel"):
                yield Label("Model Config", classes="panel-heading")
                model_options = [
                    (name, name) for name in self._model_configs
                ] or [("(none)", "(none)")]
                yield Select(model_options, id="model-select", prompt="Select model")

                yield Label("Stage", classes="panel-heading")
                stage_options = [
                    (name, name) for name in self._stage_configs
                ] or [("(none)", "(none)")]
                yield Select(stage_options, id="stage-select", prompt="Select stage")

                yield Label("Mode", classes="panel-heading")
                yield Checkbox("Sweep mode (comma-separated values)", id="sweep-toggle")
                yield Checkbox(
                    "Stage chain: d0 → d1 → d2",
                    id="chain-toggle",
                )

                yield Label("Overrides", classes="panel-heading")
                yield Label("learning_rate", classes="field-label")
                yield Input(
                    placeholder="e.g. 5e-5",
                    id="lr-input",
                    value="",
                )
                yield Label("max_length", classes="field-label")
                yield Input(
                    placeholder="e.g. 1024",
                    id="maxlen-input",
                    value="",
                )
                yield Label("model.scheduler", classes="field-label")
                yield Select(
                    _SCHEDULER_OPTIONS,
                    id="scheduler-select",
                    prompt="(default)",
                    allow_blank=True,
                )
                yield Label("model.loss_weight_type", classes="field-label")
                yield Select(
                    _LOSS_WEIGHT_OPTIONS,
                    id="loss-weight-select",
                    prompt="(default)",
                    allow_blank=True,
                )
                yield Label("model.time_epsilon", classes="field-label")
                yield Input(placeholder="e.g. 0.001", id="epsilon-input", value="")
                yield Label("training.batch_size", classes="field-label")
                yield Input(placeholder="e.g. 64", id="batch-input", value="")

                yield Button("▶  Launch", id="launch-btn", variant="success")

            with Vertical(id="right-panel"):
                yield Label("Config Preview", classes="panel-heading")
                yield Static("", id="preview-static", markup=False)
                yield Label("", id="validation-label", classes="validation-label")

        yield Footer()

    # ------------------------------------------------------------------
    # Reactive helpers
    # ------------------------------------------------------------------

    def _collect_overrides(self) -> dict[str, str]:
        """Collect current override values from widgets."""
        overrides: dict[str, str] = {}

        def _get_input(widget_id: str) -> str:
            try:
                return self.query_one(f"#{widget_id}", Input).value
            except NoMatches:
                return ""

        def _get_select(widget_id: str) -> str:
            try:
                val = self.query_one(f"#{widget_id}", Select).value
                return str(val) if val and val != Select.BLANK else ""
            except NoMatches:
                return ""

        lr = _get_input("lr-input")
        if lr:
            overrides["training.learning_rate"] = lr
        maxlen = _get_input("maxlen-input")
        if maxlen:
            overrides["dataset.max_length"] = maxlen
        scheduler = _get_select("scheduler-select")
        if scheduler:
            overrides["model.scheduler"] = scheduler
        loss_weight = _get_select("loss-weight-select")
        if loss_weight:
            overrides["model.loss_weight_type"] = loss_weight
        epsilon = _get_input("epsilon-input")
        if epsilon:
            overrides["model.time_epsilon"] = epsilon
        batch = _get_input("batch-input")
        if batch:
            overrides["training.batch_size"] = batch

        return overrides

    def _get_selected_stage(self) -> str:
        try:
            val = self.query_one("#stage-select", Select).value
            return str(val) if val and val != Select.BLANK else "d0"
        except NoMatches:
            return "d0"

    def _get_sweep_mode(self) -> bool:
        try:
            return self.query_one("#sweep-toggle", Checkbox).value
        except NoMatches:
            return False

    def _get_chain_mode(self) -> bool:
        try:
            return self.query_one("#chain-toggle", Checkbox).value
        except NoMatches:
            return False

    def _validate_overrides(self, overrides: dict[str, str]) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []
        float_fields = {
            "training.learning_rate",
            "model.time_epsilon",
        }
        int_fields = {
            "dataset.max_length",
            "training.batch_size",
        }
        valid_schedulers = {"linear", "cosine"}
        valid_loss_weights = {"uniform", "scheduler"}

        for key, val in overrides.items():
            if key in float_fields:
                for v in (v.strip() for v in val.split(",") if v.strip()):
                    try:
                        float(v)
                    except ValueError:
                        errors.append(f"{key}: '{v}' is not a valid float")
            elif key in int_fields:
                for v in (v.strip() for v in val.split(",") if v.strip()):
                    try:
                        int(v)
                    except ValueError:
                        errors.append(f"{key}: '{v}' is not a valid integer")
            elif key == "model.scheduler":
                for v in (v.strip() for v in val.split(",") if v.strip()):
                    if v not in valid_schedulers:
                        errors.append(
                            f"model.scheduler: '{v}' not in {valid_schedulers}"
                        )
            elif key == "model.loss_weight_type":
                for v in (v.strip() for v in val.split(",") if v.strip()):
                    if v not in valid_loss_weights:
                        errors.append(
                            f"model.loss_weight_type: '{v}' not in {valid_loss_weights}"
                        )
        return errors

    def _update_preview(self) -> None:
        """Recompute and display the config preview."""
        stage = self._get_selected_stage()
        overrides = self._collect_overrides()
        errors = self._validate_overrides(overrides)

        try:
            validation_label = self.query_one("#validation-label", Label)
        except NoMatches:
            return

        if errors:
            validation_label.update("⚠ " + "  |  ".join(errors))
            validation_label.add_class("has-error")
        else:
            validation_label.update("")
            validation_label.remove_class("has-error")

        try:
            preview_static = self.query_one("#preview-static", Static)
        except NoMatches:
            return

        # Build display text
        sweep_mode = self._get_sweep_mode()
        chain_mode = self._get_chain_mode()

        lines: list[str] = []

        if chain_mode:
            # Stage chain preview
            lines.append("# Stage chain: d0 → d1 → d2\n")
            for s in _STAGE_CHAIN_STAGES:
                override_str = build_override_string(s, overrides)
                lines.append(f"# --- {s} ---")
                lines.append(f"# override: {override_str}\n")
                merged = resolve_config_preview(s, overrides)
                lines.append(yaml.dump(merged, default_flow_style=False, sort_keys=True))
        elif sweep_mode:
            # Show grid summary
            sweep_params: dict[str, list[str]] = {}
            for key, val in overrides.items():
                values = [v.strip() for v in val.split(",") if v.strip()]
                if values:
                    sweep_params[key] = values
            grid = build_multirun_overrides(stage, sweep_params)
            lines.append(f"# Sweep: {len(grid)} runs (stage={stage})\n")
            for i, override_str in enumerate(grid[:10], 1):
                lines.append(f"  [{i:2d}] {override_str}")
            if len(grid) > 10:
                lines.append(f"  ... and {len(grid) - 10} more")
            lines.append("\n# First run preview:")
            if grid:
                first_overrides: dict[str, str] = {}
                for part in grid[0].split():
                    if "=" in part and not part.startswith("stage="):
                        k, v = part.split("=", 1)
                        first_overrides[k] = v
                merged = resolve_config_preview(stage, first_overrides)
                lines.append(yaml.dump(merged, default_flow_style=False, sort_keys=True))
        else:
            # Single run preview
            override_str = build_override_string(stage, overrides)
            lines.append(f"# override: {override_str}\n")
            merged = resolve_config_preview(stage, overrides)
            lines.append(yaml.dump(merged, default_flow_style=False, sort_keys=True))

        preview_static.update("\n".join(lines))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._update_preview()

    @on(Select.Changed)
    def on_select_changed(self, _event: Select.Changed) -> None:
        self._update_preview()

    @on(Input.Changed)
    def on_input_changed(self, _event: Input.Changed) -> None:
        self._update_preview()

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, _event: Checkbox.Changed) -> None:
        self._update_preview()

    @on(Button.Pressed, "#launch-btn")
    async def on_launch(self, _event: Button.Pressed) -> None:
        """Handle Launch button press."""
        stage = self._get_selected_stage()
        overrides = self._collect_overrides()
        errors = self._validate_overrides(overrides)
        if errors:
            try:
                label = self.query_one("#validation-label", Label)
                label.update("⚠ Fix errors before launching: " + " | ".join(errors))
                label.add_class("has-error")
            except NoMatches:
                pass
            return

        chain_mode = self._get_chain_mode()
        sweep_mode = self._get_sweep_mode()

        if chain_mode:
            override_strings = [
                build_override_string(s, overrides) for s in _STAGE_CHAIN_STAGES
            ]
            await self._runner.submit_queue(override_strings)
        elif sweep_mode:
            sweep_params: dict[str, list[str]] = {}
            for key, val in overrides.items():
                values = [v.strip() for v in val.split(",") if v.strip()]
                if values:
                    sweep_params[key] = values
            override_strings = build_multirun_overrides(stage, sweep_params)
            await self._runner.submit_queue(override_strings)
        else:
            override_str = build_override_string(stage, overrides)
            await self._runner.submit(override_str)

        # Switch to monitor screen
        self.app.action_show_monitor()

    def action_toggle_mode(self) -> None:
        try:
            cb = self.query_one("#sweep-toggle", Checkbox)
            cb.value = not cb.value
        except NoMatches:
            pass

    BINDINGS = [
        Binding("m", "app.show_monitor", "Monitor"),
    ]


# ---------------------------------------------------------------------------
# MonitorScreen
# ---------------------------------------------------------------------------


class MonitorScreen(Screen):
    """Live log tail + job queue status."""

    TITLE = "Monitor"

    BINDINGS = [
        Binding("s", "app.show_setup", "Setup"),
        Binding("c", "cancel_job", "Cancel"),
    ]

    def __init__(self, runner: Runner) -> None:
        super().__init__()
        self._runner = runner
        self._selected_job: Job | None = None
        self._tail_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="monitor-body"):
            with Vertical(id="queue-panel"):
                yield Label("Job Queue", classes="panel-heading")
                yield Static("", id="queue-static", markup=False)
                yield Label("Completed Checkpoints", classes="panel-heading")
                yield Static("", id="checkpoints-static", markup=False)

            with Vertical(id="log-panel"):
                yield Label("Live Log", classes="panel-heading")
                with ScrollableContainer(id="log-scroll"):
                    yield Log(id="log-widget", auto_scroll=True)

        yield Footer()

    def on_mount(self) -> None:
        self._refresh_queue()
        self._refresh_checkpoints()
        self.set_interval(1.0, self._refresh_queue)
        self._start_tail()

    def _refresh_queue(self) -> None:
        """Refresh the job queue display."""
        try:
            queue_static = self.query_one("#queue-static", Static)
        except NoMatches:
            return

        jobs = self._runner.queue
        if not jobs:
            queue_static.update("(no jobs this session)")
            return

        lines: list[str] = []
        for job in jobs:
            marker = "▶ " if job == self._runner.active_job else "  "
            state_str = job.state.name
            short_id = job.run_id[:20]
            short_override = job.override_string[:40]
            lines.append(f"{marker}[{state_str:9s}] {short_id} {short_override}")
        queue_static.update("\n".join(lines))

        # If no tail is running but there's a new active job, start tailing
        if self._runner.active_job and (
            self._tail_worker is None or self._tail_worker.is_finished
        ):
            self._start_tail()

    def _refresh_checkpoints(self) -> None:
        """List completed checkpoint directories."""
        try:
            checkpoints_static = self.query_one("#checkpoints-static", Static)
        except NoMatches:
            return

        if not _WEIGHTS_CHECKPOINTS.exists():
            checkpoints_static.update("(none)")
            return

        dirs = sorted(_WEIGHTS_CHECKPOINTS.iterdir())
        if not dirs:
            checkpoints_static.update("(none)")
            return

        checkpoints_static.update(
            "\n".join(f"  {d.name}" for d in dirs if d.is_dir())
        )

    def _start_tail(self) -> None:
        """Start tailing the active job's log."""
        active = self._runner.active_job
        if active is None:
            return
        if self._tail_worker and not self._tail_worker.is_finished:
            return
        self._tail_worker = self.run_worker(self._tail_job(active), exclusive=True)

    async def _tail_job(self, job: Job) -> None:
        """Worker coroutine: tail job log and append lines to Log widget."""
        try:
            log_widget = self.query_one("#log-widget", Log)
        except NoMatches:
            return

        log_widget.clear()
        log_widget.write_line(f"[Tailing: {job.run_id}]")

        try:
            async for line in self._runner.tail(job):
                try:
                    log_widget = self.query_one("#log-widget", Log)
                except NoMatches:
                    return
                log_widget.write_line(line.rstrip())
        except Exception as exc:
            try:
                log_widget = self.query_one("#log-widget", Log)
                log_widget.write_line(f"[tail error: {exc}]")
            except NoMatches:
                pass

    async def action_cancel_job(self) -> None:
        """Cancel the active job."""
        await self._runner.cancel()
        self._refresh_queue()

    def on_unmount(self) -> None:
        if self._tail_worker and not self._tail_worker.is_finished:
            self._tail_worker.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    background: $surface;
}

Header {
    background: $primary;
}

#setup-body, #monitor-body {
    layout: horizontal;
    height: 1fr;
}

#left-panel, #queue-panel {
    width: 40%;
    border: solid $primary-darken-2;
    padding: 1 2;
    overflow-y: auto;
}

#right-panel, #log-panel {
    width: 60%;
    border: solid $primary-darken-2;
    padding: 1 2;
    overflow-y: auto;
}

.panel-heading {
    text-style: bold;
    color: $accent;
    margin-top: 1;
    margin-bottom: 0;
}

.field-label {
    margin-top: 1;
    color: $text-muted;
}

Input {
    margin-bottom: 0;
}

Select {
    margin-bottom: 0;
}

#launch-btn {
    margin-top: 2;
    width: 100%;
}

#preview-static {
    width: 100%;
    color: $text;
}

.validation-label {
    color: $warning;
    margin-top: 1;
}

.validation-label.has-error {
    color: $error;
}

#log-scroll {
    height: 1fr;
    overflow-y: auto;
}

#log-widget {
    height: 1fr;
}

#queue-static, #checkpoints-static {
    color: $text;
    width: 100%;
}
"""


class DialSFTApp(App):
    CSS = _CSS
    BINDINGS = [
        Binding("s", "show_setup", "Setup"),
        Binding("m", "show_monitor", "Monitor"),
        Binding("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        self.runner = Runner()
        self.push_screen(SetupScreen(self.runner))

    def action_show_setup(self) -> None:
        self.push_screen(SetupScreen(self.runner))

    def action_show_monitor(self) -> None:
        self.push_screen(MonitorScreen(self.runner))


def run() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    DialSFTApp().run()
