# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["ComputeGraphNetwork", "-h"])

    assert ret.success


def test_compute_graph_network(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_matrix = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_folder = os.path.join(get_home(), "data/graph_network/")

    ret = script_runner.run([
        "ComputeGraphNetwork",
        "--in-dataset", in_matrix,
        "--out-folder", out_folder,
        "--id-column", "subjectkey",
        "--desc-columns", 1, "-f"]
    )

    assert ret.success
