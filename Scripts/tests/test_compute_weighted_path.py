# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_compute_weighted_path import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_compute_weighted_path():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")
    data_label = os.path.join(get_home(), "data/labels.xlsx")
    out_folder = os.path.join(get_home(), "data/weighted_path/")

    ret = runner.invoke(
        app,
        [
            "--in-graph",
            in_graph,
            "--data-for-label",
            data_label,
            "--id-column",
            "subjectkey",
            "--label-name",
            "diagnosis",
            "--out-folder",
            out_folder,
            "--iterations",
            10,
            "--processes",
            1,
            "-f",
        ],
    )

    assert ret.exit_code == 0
