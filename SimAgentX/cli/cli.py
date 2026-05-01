import typer
from app.main import main

app = typer.Typer()

@app.command()
def run():
    main()

if __name__ == "__main__":
    app()
