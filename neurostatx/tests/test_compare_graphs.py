# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from neurostatx.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["CompareGraphs", "-h"])

    assert ret.success


def test_compute_graph_network(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")
    out_folder = os.path.join(get_home(), "data/compare_graph/")

    ret = script_runner.run([
        "CompareGraphs",
        "--in-graph1", in_graph,
        "--weight", "membership",
        "--percentile", 80,
        "--in-graph2", in_graph,
        "--out-folder", out_folder,
        "-f", "-v"]
    )

    assert ret.success
