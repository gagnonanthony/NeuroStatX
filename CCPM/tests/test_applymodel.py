#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run(["ApplyModel", "-h"])

    assert ret.success


def test_execution_apply_model(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), "data/factor_data.xlsx")
    out_folder = os.path.join(get_home(), "data/apply_model/")
    model = os.path.join(get_home(), "data/EFA_model.pkl")

    ret = script_runner.run([
        "ApplyModel",
        "--in-dataset", in_dataset,
        "--out-folder", out_folder,
        "--desc-columns", 1,
        "--id-column", "subjectkey",
        "--model", model,
        "-f", "-v", "-s"]
    )

    assert ret.success
