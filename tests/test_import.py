"""
Basic tests verifying that packages and submodules can be imported.
"""

import ssp_framework


def test_version() -> None:
    assert ssp_framework.__version__ == "0.1.0.dev0"


def test_imports() -> None:
    from ssp_framework import core
    from ssp_framework import data
    from ssp_framework import supervised
    from ssp_framework import rl
    from ssp_framework import evaluation
    from ssp_framework import utils

    assert core is not None
    assert data is not None
    assert supervised is not None
    assert rl is not None
    assert evaluation is not None
    assert utils is not None
