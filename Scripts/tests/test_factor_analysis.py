#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_factor_analysis import app


download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ['--help'])

    assert ret.exit_code == 0


def test_execution_factor_analysis():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/factor_data.xlsx')
    out_folder = os.path.join(get_home(), 'data/factor_results/')

    ret = runner.invoke(app, ['--in-dataset', in_dataset, '--out-folder', out_folder,
                        '--id-column', 'subjectkey', '-f'])

    assert ret.exit_code == 0
