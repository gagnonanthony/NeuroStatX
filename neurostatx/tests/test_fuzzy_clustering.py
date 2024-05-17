#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from neurostatx.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["FuzzyClustering", "-h"])

    assert ret.success


def test_execution_fuzzy(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), "data/clustering_data.xlsx")
    out_folder = os.path.join(get_home(), "data/Fuzzy_Clustering/")

    ret = script_runner.run([
        "FuzzyClustering",
        "--out-folder", out_folder,
        "--in-dataset", in_dataset,
        "--desc-columns", 4,
        "--id-column", "subjectkey",
        "--k", 3,
        "--parallelplot",
        "--radarplot",
        "-f"]
    )

    assert ret.success
