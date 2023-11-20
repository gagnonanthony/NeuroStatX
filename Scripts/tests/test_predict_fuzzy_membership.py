#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_predict_fuzzy_membership import app

download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_execution_predict_fuzzy():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), "data/clustering_data.xlsx")
    in_centroids = os.path.join(
        get_home(),
        "data/Fuzzy_Clustering/CENTROIDS/clusters_centroids_3.xlsx")
    out_folder = os.path.join(get_home(), "data/Predicted_Clustering/")

    ret = runner.invoke(
        app,
        [
            "--out-folder",
            out_folder,
            "--in-cntr",
            in_centroids,
            "--in-dataset",
            in_dataset,
            "--desc-columns",
            1,
            "--id-column",
            "subjectkey",
            "-f",
        ],
    )

    assert ret.exit_code == 0
