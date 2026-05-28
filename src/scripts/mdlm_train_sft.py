import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(rich_markup_mode="rich")


def _show_config(cfg) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("[Paths]", "")
    table.add_row("  train data",   cfg.TRAIN_DATA_LOAD_PATH)
    table.add_row("  test data",    cfg.TEST_DATA_LOAD_PATH)
    table.add_row("  model load",   cfg.TRAIN_MODEL_LOAD_PATH)
    table.add_row("  model save",   cfg.TRAIN_MODEL_SAVE_PATH)
    table.add_row("[Dataset]", "")
    table.add_row("  num_samples",  str(cfg.num_samples))
    table.add_row("  num_workers",  str(cfg.num_workers))
    table.add_row("  max_length",   str(cfg.max_length))
    table.add_row("[Training]", "")
    table.add_row("  epochs",       str(cfg.num_epochs))
    table.add_row("  batch_size",   str(cfg.batch_size))
    table.add_row("  lr",           str(cfg.learning_rate))
    table.add_row("  time_epsilon", str(cfg.time_epsilon))
    table.add_row("  loss_weight",  cfg.loss_weight_type)
    table.add_row("[Reporting]", "")
    table.add_row("  eval_strategy", cfg.eval_strategy)
    table.add_row("  save_strategy", cfg.save_strategy)
    table.add_row("  report_to",     ", ".join(cfg.report_to))

    console.print(Panel(table, title="[bold cyan]Training Config[/]", expand=False))


@app.command()
def main(
    # --- paths ---
    train_data: Annotated[str, typer.Option(
        prompt="Train data path",
        help="Path to training dataset on disk",
    )] = "./datasets/base/writingprompts_train",

    test_data: Annotated[str, typer.Option(
        prompt="Test data path",
        help="Path to test dataset on disk",
    )] = "./datasets/base/writingprompts_test",

    model_load: Annotated[str, typer.Option(
        prompt="Model load path",
        help="Path to pretrained weights",
    )] = "./weights/base",

    model_save: Annotated[str, typer.Option(
        prompt="Model save path",
        help="Where to write checkpoints",
    )] = "./weights/checkpoints",

    # --- dataset ---
    num_samples: Annotated[int, typer.Option(
        prompt="Num samples (train + test)",
        help="How many examples to load from each split",
    )] = 1000,

    num_workers: Annotated[int, typer.Option(
        prompt="Dataloader num workers",
    )] = 4,

    max_length: Annotated[int, typer.Option(
        prompt="Max token length",
    )] = 512,

    # --- training ---
    num_epochs: Annotated[int, typer.Option(
        prompt="Num epochs",
    )] = 2,

    batch_size: Annotated[int, typer.Option(
        prompt="Batch size",
    )] = 16,

    learning_rate: Annotated[float, typer.Option(
        prompt="Learning rate",
    )] = 2e-5,

    time_epsilon: Annotated[float, typer.Option(
        prompt="Time epsilon (MDLM)",
    )] = 0.001,

    loss_weight_type: Annotated[str, typer.Option(
        prompt="Loss weight type",
        help="uniform | linear | ...",
    )] = "uniform",

    # --- reporting ---
    eval_strategy: Annotated[str, typer.Option(
        prompt="Eval strategy",
        help="epoch | steps",
    )] = "epoch",

    save_strategy: Annotated[str, typer.Option(
        prompt="Save strategy",
        help="epoch | steps",
    )] = "epoch",

    report_to: Annotated[str, typer.Option(
        prompt="Report to (comma-separated)",
        help="e.g. tensorboard,wandb",
    )] = "tensorboard",
):
    """
    [bold]Run MDLM SFT training.[/]
    Run with no arguments to be walked through all options interactively.
    All options can also be passed as flags to skip prompts.
    """
    from src.mdlm.mdlm_train_sft import TrainingConfig, run_training
    cfg = TrainingConfig(
        TRAIN_DATA_LOAD_PATH=train_data,
        TEST_DATA_LOAD_PATH=test_data,
        TRAIN_MODEL_LOAD_PATH=model_load,
        TRAIN_MODEL_SAVE_PATH=model_save,
        num_samples=num_samples,
        num_workers=num_workers,
        max_length=max_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        time_epsilon=time_epsilon,
        loss_weight_type=loss_weight_type,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        report_to=[r.strip() for r in report_to.split(",")],
    )

    _show_config(cfg)

    if not typer.confirm("\nProceed with training?", default=True):
        raise typer.Abort()

    run_training(cfg)


if __name__ == "__main__":
    app()