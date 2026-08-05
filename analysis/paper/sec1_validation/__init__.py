"""ADoNIS-vs-ACHILLES paper figures (arXiv:2508.19213).

ONE way to run them:  python -m analysis.paper.sec1_validation.make [names] [--all]
The default full build skips heavy opt-out specs (fig13: live MC, no cache); --all or naming it includes it.
Each figure is a YAML spec in this directory; helper.py renders it (spec['render'] picks the function);
make.py is the only entry point.  Nothing else draws these figures.
"""
