# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["LogisticRegression", "-h"])

    assert ret.success


def test_logistic_regression(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_with_attributes.gml")
    out_folder = os.path.join(get_home(), "data/LogisticRegression_results")

    ret = script_runner.run([
        "LogisticRegression",
        "--in-graph", in_graph,
        "--out-folder", out_folder,
        "--attributes", "diagnosis",
        "--covariates", "age",
        "--covariates", "iq",
        "--permutations", 10,
        "--plot-distributions",
        "-f"]
    )

    assert ret.success
