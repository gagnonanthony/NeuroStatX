#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from neurostatx.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["PredictFuzzyMembership", "-h"])

    assert ret.success


def test_execution_predict_fuzzy(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), "data/clustering_data.xlsx")
    in_centroids = os.path.join(
        get_home(),
        "data/Fuzzy_Clustering/CENTROIDS/clusters_centroids_3.xlsx")
    out_folder = os.path.join(get_home(), "data/Predicted_Clustering/")

    ret = script_runner.run([
        "PredictFuzzyMembership",
        "--out-folder", out_folder,
        "--in-cntr", in_centroids,
        "--in-dataset", in_dataset,
        "--desc-columns", 4,
        "--id-column", "subjectkey",
        "--parallelplot",
        "--radarplot",
        "-f", "-v", "-s"]
    )

    assert ret.success
