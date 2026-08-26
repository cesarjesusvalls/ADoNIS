"""Shim kept so the S3DF submitters keep working; the tool is analysis.oracle_tools.combine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.oracle_tools.combine import combine

if __name__ == "__main__":
    combine(sys.argv[1], sys.argv[2])
