# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_compare_graphs import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_compute_graph_network():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")
    in_mat = os.path.join(get_home(), "data/clusters_membership_3.npy")
    out_folder = os.path.join(get_home(), "data/compare_graph/")

    ret = runner.invoke(
        app,
        [
            "--in-graph1",
            in_graph,
            "--in-matrix",
            in_mat,
            "--percentile",
            80,
            "--in-graph2",
            in_graph,
            "--out-folder",
            out_folder,
            "-f",
        ],
    )

    assert ret.exit_code == 0
