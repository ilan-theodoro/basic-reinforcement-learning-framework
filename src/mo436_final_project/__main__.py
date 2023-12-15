"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Reinforcement Learning course's final project.."""


if __name__ == "__main__":
    main(prog_name="mo436_final_project")  # pragma: no cover
