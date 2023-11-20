#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_compare_clustering import app

download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_execution_compare_clustering():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset1 = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    in_dataset2 = os.path.join(get_home(), "data/clusters_membership_3.xlsx")
    out_folder = os.path.join(get_home(), "data/ARI/")

    ret = runner.invoke(
        app,
        [
            "--in-dataset",
            in_dataset1,
            "--in-dataset",
            in_dataset2,
            "--desc-columns",
            1,
            "--id-column",
            "subjectkey",
            "--out-folder",
            out_folder,
            "-f",
        ],
    )

    assert ret.exit_code == 0
