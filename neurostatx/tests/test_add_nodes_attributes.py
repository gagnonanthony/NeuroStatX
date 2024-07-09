# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from neurostatx.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["AddNodesAttributes", "-h"])

    assert ret.success


def test_set_nodes_attributes(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")
    in_df = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_file = os.path.join(get_home(), "data/graph_with_attributes.gml")

    ret = script_runner.run([
        "AddNodesAttributes",
        "--in-graph", in_graph,
        "--in-dataset", in_df,
        "--labels", "gestage",
        "--labels", "age",
        "--labels", "iq",
        "--id-column", "subjectkey",
        "--out-file", out_file,
        "-f", "-v"]
    )

    assert ret.success
