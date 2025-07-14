"""Formatting pipeline, it will run formatting and linting tools to ensure code property."""

import subprocess
import sys


def run_command(command: str) -> int:
    try:
        result = subprocess.run(command, check=True, shell=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)


def main():
    # Clean imports
    print("Running isort...")
    run_command("python3 -m isort src/ scripts/.")
    
    # Code-style formatting
    print("Running black...")
    run_command("python3 -m black src/ scripts/")

    # Linters
    print("Running flake8...")
    run_command("python3 -m flake8 --toml-config pyproject.toml --ignore E501,E203,W503 src/ scripts/.")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
