"""Force-link the cascade interaction registry into the ACHILLES binaries.

The registry is populated by static registration inside AchillesCascadeInteractions, which the
linker drops because nothing references it, and by a Plugin::Manager that CascadeMain never
instantiates.  Either omission leaves the registry empty and the cascade driver aborts on the first
interaction lookup.  This adds the link dependency (with --no-as-needed, so the linker keeps it)
and constructs the manager.

    python3 patch_cascade_registry.py <achilles-source-root> [--main]

--main patches the main `achilles` binary as well as the standalone `achilles-cascade`, which is
what in-event FSI needs.  Idempotent: applying it twice is a no-op.
"""
import sys
from pathlib import Path

STANDALONE = [
    ("PUBLIC event_gen docopt::docopt cmake_git_version_tracking)",
     "PUBLIC event_gen docopt::docopt cmake_git_version_tracking dl plugin_manager "
     "AchillesCascadeInteractions)"),
    ("    list(APPEND achilles_targets achilles-cascade)",
     '    target_link_options(achilles-cascade PRIVATE "LINKER:--no-as-needed")\n'
     "    list(APPEND achilles_targets achilles-cascade)"),
]

MAIN = [
    ("PUBLIC event_gen docopt::docopt dl cmake_git_version_tracking plugin_manager)",
     "PUBLIC event_gen docopt::docopt dl cmake_git_version_tracking plugin_manager "
     "AchillesCascadeInteractions)"),
    ("list(APPEND achilles_targets achilles plugin_manager)",
     'target_link_options(achilles PRIVATE "LINKER:--no-as-needed")\n'
     "    list(APPEND achilles_targets achilles plugin_manager)"),
]

CASCADE_MAIN = [
    ('#include "Achilles/Version.hh"',
     '#include "Achilles/Version.hh"\n#include "Plugins/Manager/PluginManager.hh"'),
    ("    achilles::CascadeTest::RunCascade(runcard);",
     "    achilles::Plugin::Manager plugin_manager;\n"
     "    achilles::CascadeTest::RunCascade(runcard);"),
]


def apply(path, edits):
    src = path.read_text()
    for old, new in edits:
        if new in src:
            continue
        if old not in src:
            raise SystemExit(f"{path}: expected text not found -- upstream has changed:\n  {old[:70]}")
        src = src.replace(old, new)
    path.write_text(src)


def main(argv):
    root = Path(argv[1] if len(argv) > 1 else "/achilles_src")
    want_main = "--main" in argv
    cmake = root / "src" / "Achilles" / "CMakeLists.txt"
    apply(cmake, STANDALONE + (MAIN if want_main else []))
    apply(root / "src" / "Achilles" / "CascadeMain.cc", CASCADE_MAIN)
    print(f"patched {cmake.name} ({'main + standalone' if want_main else 'standalone'}) and CascadeMain.cc")


if __name__ == "__main__":
    main(sys.argv)
