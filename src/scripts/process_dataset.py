import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel

from src.dataset_load import DatasetProcessingConfig, _process_and_save_datasets

console = Console()
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def main(
    dataset_name: Annotated[str, typer.Option(
        prompt="HuggingFace dataset name",
        help="e.g. euclaise/writingprompts",
    )] = "euclaise/writingprompts",

    model_name: Annotated[str, typer.Option(
        prompt="Tokenizer/model name",
        help="HuggingFace model to load tokenizer from",
    )] = "EleutherAI/pythia-70m",

    output_dir: Annotated[str, typer.Option(
        prompt="Output base directory",
        help="Where processed splits will be saved",
    )] = "./datasets/base",
):
    """Process a HuggingFace dataset — tokenize, rename columns, assign IDs, save to disk."""
    slug = dataset_name.split("/")[-1]
    cfg = DatasetProcessingConfig(
        dataset_name=dataset_name,
        split_save_paths={
            "train":      f"{output_dir}/{slug}_train",
            "validation": f"{output_dir}/{slug}_eval",
            "test":       f"{output_dir}/{slug}_test",
        },
    )

    console.print(Panel(
        f"[bold]Dataset:[/]    {dataset_name}\n"
        f"[bold]Model:[/]      {model_name}\n"
        f"[bold]Output dir:[/] {output_dir}",
        title="[bold cyan]Config[/]",
        expand=False,
    ))

    if not typer.confirm("Proceed?", default=True):
        raise typer.Abort()

    from src.ar.ar_baseline import setup_model_and_tokenizer
    _, tokenizer = setup_model_and_tokenizer(model_name=model_name)
    _process_and_save_datasets(cfg, tokenizer)


if __name__ == "__main__":
    app()