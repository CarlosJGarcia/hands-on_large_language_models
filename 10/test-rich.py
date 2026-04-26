from rich.table import Table
from rich.console import Console

console = Console()

table = Table(title="Model Training Benchmarks")

table.add_column("Epoch", justify="right", style="cyan")
table.add_column("Loss", style="magenta")
table.add_column("Accuracy", justify="right", style="green")

# You can update this inside your training loop
table.add_row("1", "0.452", "82%")
table.add_row("2", "0.312", "89%")

console.print(table)

console.print("This is [bold red]Red and Bold[/bold red]!")
console.print(":rocket: [green]Training started successfully on your RTX 5060 Ti![/green]")
print("Texto normal")
console.print("[red]Texto rojo[/red]")
print("Fin del programa.")
