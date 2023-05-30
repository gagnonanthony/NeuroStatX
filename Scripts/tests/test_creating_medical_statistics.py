# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_creating_medical_statistics import app


download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ['--help'])

    assert ret.exit_code == 0


def test_execution_medical_stats():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/demographics.xlsx')
    out_table = os.path.join(get_home(), 'data/table.xlsx')

    ret = runner.invoke(app, ['--in-dataset', in_dataset, '--output', out_table,
                        '--id-column', 'subid', '-r', 'Sex', '-r', 'Age', '-r', 'IQ',
                        '-c', 'Sex', '-n', 'Sex', '-n', 'Age', '-n', 'Quotient',
                        '--apply_yes_or_no', '-f'])

    assert ret.exit_code == 0
