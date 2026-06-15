"""ADoNIS analysis driver: render an ACH/ADO ratio+chi2 figure from a YAML config.

Usage: python -u scripts/adonis_analyze.py configs/ana_cc1pi.yaml
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adonis.workflow.config import load_analysis_config
from adonis.workflow.analyze import run_analysis

if __name__ == "__main__":
    cfg = load_analysis_config(sys.argv[1])
    run_analysis(cfg)
