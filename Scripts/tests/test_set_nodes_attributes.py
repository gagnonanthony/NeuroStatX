# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_set_nodes_attributes import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_set_nodes_attributes():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")
    in_df = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_file = os.path.join(get_home(), "data/graph_with_attributes.gexf")

    ret = runner.invoke(
        app,
        [
            "--in-graph",
            in_graph,
            "--in-dataset",
            in_df,
            "--labels",
            "Cluster #1",
            "--id-column",
            "subjectkey",
            "--out-file",
            out_file,
            "-f",
        ],
    )

    assert ret.exit_code == 0
