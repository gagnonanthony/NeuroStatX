# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_plsr import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_plsr():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_with_attributes.gexf")
    out_folder = os.path.join(get_home(), "data/PLSR_results")

    ret = runner.invoke(
        app,
        [
            "--in-graph",
            in_graph,
            "--out-folder",
            out_folder,
            "--attributes",
            "gestage",
            "--attributes",
            "age",
            "--attributes",
            "iq",
            "--permutations",
            100,
            "-f",
        ],
    )

    assert ret.exit_code == 0
