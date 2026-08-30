"""probe names are honest, everywhere, permanently.

Two bug classes are pinned dead here:

  1. "weak" as a probe value.  A neutral current is every bit as weak as a charged one, so the name
     could not survive NC without meaning "CC except when it doesn't".  Renamed to "CC" with NO alias
     and NO back-compat mapping -- a mapping is how the lie survives.
  2. The manifest disagreeing with the config that wrote it.  generate_bank hardcoded probe="ee" in
     the EM branch while GenConfig.probe was "EM"; the manifest now comes from cfg.probe, so the two
     cannot diverge again.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

RETIRED = ("weak", "ee")
_SEARCH_DIRS = ("adonis", "configs", "analysis", "tests")
_SKIP = {"test_probe_naming.py"}


def _py_offences(path, text):
    """A retired name used as a probe VALUE in Python: probe="weak", probe='ee', ("weak", ...)."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0]
        for name in RETIRED:
            if re.search(rf'probe\s*=\s*["\']{name}["\']', code) or \
               re.search(rf'["\']{name}["\']\s*:\s*\(', code):
                out.append(f"{path}:{i}: {line.strip()}")
    return out


def test_no_retired_probe_value_in_python():
    bad = []
    for d in _SEARCH_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if p.name in _SKIP:
                continue
            bad += _py_offences(p.relative_to(ROOT), p.read_text())
    assert not bad, "retired probe values still used:\n" + "\n".join(bad)


def test_no_retired_probe_value_in_bank_configs():
    bad = []
    for p in sorted((ROOT / "configs" / "banks").glob("*.yaml")):
        probe = (yaml.safe_load(p.read_text()) or {}).get("probe")
        if probe in RETIRED:
            bad.append(f"{p.relative_to(ROOT)}: probe: {probe}")
    assert not bad, "retired probe values in bank configs:\n" + "\n".join(bad)


def test_every_bank_config_declares_a_known_probe():
    from adonis.workflow.config import PROBES
    cfgs = sorted((ROOT / "configs" / "banks").glob("*.yaml"))
    assert len(cfgs) >= 12, f"expected the 12 known bank configs, found {len(cfgs)}"
    for p in cfgs:
        probe = (yaml.safe_load(p.read_text()) or {}).get("probe")
        assert probe in PROBES, f"{p.name}: probe {probe!r} not in {PROBES}"


def test_gen_config_rejects_the_old_name():
    from adonis.workflow.config import GenConfig
    with pytest.raises(ValueError, match="probe"):
        GenConfig(probe="weak", beam="spectrum")


def test_gen_config_default_probe_is_cc():
    from adonis.workflow.config import GenConfig
    assert GenConfig().probe == "CC"


def test_manifest_probe_equals_config_probe():
    """The manifest must not disagree with the config that wrote it.  Reads the literal source of the manifest dicts rather than generating a
    bank (minutes of GPU): every `probe=` in a MANIFEST construction must be `cfg.probe`.  The one
    legitimate probe= LITERAL is build_hv_sf(..., probe="EM"): the (e,e') hard-vertex records are the EM
    records by construction, independent of any config, so that call line is excluded from the scan."""
    src = (ROOT / "adonis" / "workflow" / "generate_bank.py").read_text()
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines()
                     if "build_hv_sf(" not in ln)
    literals = re.findall(r'probe\s*=\s*(["\'][^"\']*["\'])', code)
    assert not literals, f"manifest probe must come from cfg.probe, found literals: {literals}"
    assert src.count("probe=cfg.probe") >= 2, "both the hard-vertex and hadron manifests must use cfg.probe"


def test_manifests_on_disk_carry_no_retired_probe(adonis_out):
    """If the output tree is reachable, no manifest may still say weak/ee.  Skipped when it is not --
    the migration is data, and CI without the data volume must not fail on its absence."""
    stale = []
    for m in adonis_out.rglob("manifest.json"):
        try:
            probe = json.loads(m.read_text()).get("probe")
        except Exception:
            continue
        if probe in RETIRED:
            stale.append(str(m))
    assert not stale, (f"{len(stale)} manifest(s) still carry a retired probe value, so they predate "
                       f"the rename and must be regenerated:\n" + "\n".join(stale[:10]))


@pytest.fixture
def adonis_out():
    import os
    root = os.environ.get("ADONIS_OUT")
    if not root or not Path(root).is_dir():
        pytest.skip("ADONIS_OUT not set or not present (no data volume)")
    return Path(root)
