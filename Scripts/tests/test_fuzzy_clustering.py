#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_fuzzy_clustering import app

download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ['--help'])

    assert ret.exit_code == 0


def test_execution_fuzzy():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/clustering_data.xlsx')
    out_folder = os.path.join(get_home(), 'data/Fuzzy_Clustering/')

    ret = runner.invoke(app, ['--out-folder', out_folder, '--in-dataset', in_dataset,
                              '--desc-columns', 1, '--id-column', 'subjectkey',
                              '-f'])

    assert ret.exit_code == 0