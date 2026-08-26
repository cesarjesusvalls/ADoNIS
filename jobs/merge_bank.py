"""Shim kept so the S3DF submitters keep working; the tool is adonis.workflow.merge_bank."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.workflow.merge_bank import main

if __name__ == "__main__":
    main()
