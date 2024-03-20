#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["VisualizeGraphNetwork", "-h"])

    assert ret.success


def test_visualize_graph_network(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")
    out_folder = os.path.join(get_home(), "data/Visualize_graph/")

    ret = script_runner.run([
        "VisualizeGraphNetwork",
        "--in-graph", in_graph,
        "--out-folder", out_folder,
        "--weight", "membership",
        "--label-name", "diagnosis",
        "-f", "-v", "-s"]
    )

    assert ret.success
