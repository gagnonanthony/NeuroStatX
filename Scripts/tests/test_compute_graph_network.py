# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_compute_graph_network import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_compute_graph_network():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_matrix = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_folder = os.path.join(get_home(), "data/graph_network/")

    ret = runner.invoke(
        app, ["--in-dataset", in_matrix, "--out-folder", out_folder,
              "--id-column", "subjectkey", "--desc-columns", 1, "-f"]
    )

    assert ret.exit_code == 0
