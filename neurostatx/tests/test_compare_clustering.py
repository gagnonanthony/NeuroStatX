#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from neurostatx.io.download import get_home, download_data, save_files_dict

download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["CompareClustering", "-h"])

    assert ret.success


def test_execution_compare_clustering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset1 = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    in_dataset2 = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_folder = os.path.join(get_home(), "data/ARI/")

    ret = script_runner.run([
        "CompareClustering",
        "--in-dataset", in_dataset1,
        "--in-dataset", in_dataset2,
        "--desc-columns", 1,
        "--id-column", "subjectkey",
        "--out-folder", out_folder,
        "-f", "-v", "-s"]
    )

    assert ret.success
