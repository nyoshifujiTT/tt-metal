# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guard the CPU-only model unit-test catalog against stale pytest paths.

The catalog lists literal pytest paths. If a test file is renamed or deleted and
the catalog is not updated, nothing fails locally -- the breakage only shows up
in CI, as a collection error, long after the change landed. This host-only test
re-reads the catalog and asserts every listed path still exists.
"""

import re
from pathlib import Path

import pytest

CATALOG = Path(__file__).with_name("models_cpu_only_unit_tests.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _listed_pytest_paths():
    text = CATALOG.read_text()
    # Paths are the whitespace/backslash separated tokens of the `pytest ...`
    # command blocks; keep it dependency-free (no yaml import) on purpose.
    return re.findall(r"models/\S+?\.py", text)


def test_catalog_lists_at_least_one_test():
    assert _listed_pytest_paths(), f"no pytest paths found in {CATALOG.name}"


@pytest.mark.parametrize("rel_path", _listed_pytest_paths())
def test_listed_test_file_exists(rel_path):
    assert (REPO_ROOT / rel_path).is_file(), (
        f"{CATALOG.name} lists {rel_path}, which does not exist. "
        "Update the catalog when renaming or deleting a test file."
    )
