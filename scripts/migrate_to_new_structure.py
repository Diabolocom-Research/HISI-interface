#!/usr/bin/env python3
"""
Migration script to help transition from the old ASR interface structure to the new one.

This script helps users migrate their existing code and configurations to work with
the new modular package structure.
"""

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_backup() -> Path:
    """Create a backup of the current directory."""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_asr_interface_{timestamp}")

    logger.info(f"Creating backup in {backup_dir}")

    # Copy current files to backup
    current_dir = Path(".")
    for item in current_dir.iterdir():
        if item.name not in ["backup_*", ".git", "__pycache__"]:
            if item.is_file():
                shutil.copy2(item, backup_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, backup_dir / item.name)

    logger.info(f"Backup created successfully in {backup_dir}")
    return backup_dir


def migrate_requirements() -> None:
    """Migrate requirements.txt to pyproject.toml format."""
    logger.info("Migrating requirements.txt to pyproject.toml...")

    if not Path("requirements.txt").exists():
        logger.warning("requirements.txt not found, skipping migration")
        return

    # Read existing requirements
    with open("requirements.txt") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    logger.info(f"Found {len(requirements)} requirements to migrate")

    # Note: The pyproject.toml already contains the migrated dependencies
    # This is just for reference
    logger.info("Dependencies have been migrated to pyproject.toml")
    logger.info("You can now remove requirements.txt if desired")


def migrate_imports_in_file(file_path: Path) -> bool:
    """Migrate imports in a single file."""
    if not file_path.exists() or file_path.suffix != ".py":
        return False

    try:
        with open(file_path) as f:
            content = f.read()

        original_content = content

        # Replace old imports with new ones
        replacements = [
            # Old imports to new imports
            ("from real_time_asr_backend", "from asr_interface"),
            ("import real_time_asr_backend", "import asr_interface"),
            ("from utils", "from asr_interface.utils"),
            ("import utils", "import asr_interface.utils"),
        ]

        for old_import, new_import in replacements:
            content = content.replace(old_import, new_import)

        # Update specific imports
        content = content.replace(
            "from real_time_asr_backend.real_time_asr_protocols import",
            "from asr_interface.core.protocols import",
        )
        content = content.replace(
            "from real_time_asr_backend.real_time_stream_handler import",
            "from asr_interface.handlers.stream_handler import",
        )
        content = content.replace(
            "from real_time_asr_backend.slimer_whisper_online import",
            "from asr_interface.backends.whisper_loader import",
        )

        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Migrated imports in {file_path}")
            return True

        return False

    except Exception as e:
        logger.error(f"Error migrating {file_path}: {e}")
        return False


def migrate_python_files() -> None:
    """Migrate Python files to use new import structure."""
    logger.info("Migrating Python files to new import structure...")

    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))

    migrated_count = 0
    for file_path in python_files:
        if migrate_imports_in_file(file_path):
            migrated_count += 1

    logger.info(f"Migrated {migrated_count} Python files")


def create_migration_guide() -> None:
    """Create a migration guide for users."""
    guide_content = """# Migration Guide

## Overview

The ASR Interface has been refactored to follow modern Python conventions with a modular package structure. This guide helps you migrate from the old structure to the new one.

## What Changed

### Package Structure
- Old: Flat structure with scattered modules
- New: Organized package structure under `asr_interface/`

### Import Changes
- Old: `from real_time_asr_backend import ...`
- New: `from asr_interface import ...`

### Configuration
- Old: `requirements.txt`
- New: `pyproject.toml` with uv dependency management

## Migration Steps

### 1. Install Dependencies
```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
uv pip install -e .
```

### 2. Update Your Code

#### Old Imports
```python
from real_time_asr_backend.real_time_asr_protocols import ASRProcessor, ModelLoader
from real_time_asr_backend.real_time_stream_handler import RealTimeASRHandler
from real_time_asr_backend.slimer_whisper_online import WhisperOnlineLoader
```

#### New Imports
```python
from asr_interface.core.protocols import ASRProcessor, ModelLoader
from asr_interface.handlers.stream_handler import RealTimeASRHandler
from asr_interface.backends.whisper_loader import WhisperOnlineLoader
```

### 3. Update Server Code

#### Old Server
```python
from real_time_asr_server import app
```

#### New Server
```python
from asr_interface.web.server import create_app

app = create_app()
```

### 4. CLI Usage

#### Old
```bash
python real_time_asr_server.py
```

#### New
```bash
asr-interface serve
```

## Breaking Changes

1. **Import Paths**: All imports now use the `asr_interface` package
2. **Configuration**: Use `pyproject.toml` instead of `requirements.txt`
3. **CLI**: Use the new `asr-interface` command instead of running Python files directly

## Getting Help

If you encounter issues during migration:
1. Check the backup created by the migration script
2. Review the new documentation in `docs/`
3. Open an issue on GitHub

## Rollback

If you need to rollback, you can restore from the backup created by the migration script:
```bash
# Restore from backup
cp -r backup_asr_interface_YYYYMMDD_HHMMSS/* .
```
"""

    with open("MIGRATION_GUIDE.md", "w") as f:
        f.write(guide_content)

    logger.info("Created MIGRATION_GUIDE.md")


def main() -> None:
    """Main migration function."""
    logger.info("Starting ASR Interface migration...")

    try:
        # Create backup
        backup_dir = create_backup()

        # Migrate requirements
        migrate_requirements()

        # Migrate Python files
        migrate_python_files()

        # Create migration guide
        create_migration_guide()

        logger.info("Migration completed successfully!")
        logger.info(f"Backup created in: {backup_dir}")
        logger.info("Please review MIGRATION_GUIDE.md for next steps")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
