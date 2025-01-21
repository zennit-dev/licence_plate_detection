"""Linting checks module."""

import subprocess
from dataclasses import dataclass, field
from typing import List

from src.settings import logger


@dataclass(frozen=True)
class LintCommand:
    """Lint command configuration.

    Attributes:
        description: Human-readable description of the lint command
        command: List of command arguments to execute
    """

    description: str
    command: List[str] = field(default_factory=list)

    def __init__(self, description: str, command: List[str]) -> None:
        """Initialize a LintCommand.

        Args:
            description: Human-readable description of the lint command
            command: List of command arguments to execute
        """
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "command", command)


def run_command(command: List[str], description: str) -> bool:
    """Run a shell command and print its output."""
    logger.info(f"Running {description}...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"{description} failed:")
        logger.warning(result.stdout)
        logger.warning(result.stderr)
        return False
    logger.info(f"{description} passed!")
    return True


def get_linters() -> List[LintCommand]:
    """Get all linter commands with fixing enabled where possible."""
    return [
        LintCommand(
            "Black formatter",
            ["black", "."],
        ),
        LintCommand(
            "isort",
            ["isort", "."],
        ),
        LintCommand(
            "mypy type checking",
            ["mypy", "--config-file", "tools/configs/mypy.ini", "."],
        ),
        LintCommand(
            "flake8 linting",
            ["flake8", "--config", "tools/configs/.flake8"],
        ),
    ]


def run_lint() -> bool:
    """Run all linters with automatic fixing where possible.

    Returns:
        bool: True if all linters passed, False if any failed
    """
    logger.info("Running linters...")

    linters = get_linters()
    failed = False

    for linter in linters:
        if not run_command(linter.command, linter.description):
            failed = True

    if failed:
        logger.error("Linting failed!")
        return False

    logger.success("All linters passed!")
    return True
