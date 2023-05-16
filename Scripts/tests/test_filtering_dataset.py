#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run('CCPM_filtering_dataset.py',
                            '--help')

    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/data_example.xlsx')
    out_folder = os.path.join(get_home(), 'data/Filtering/')
    ret = script_runner.run('CCPM_filtering_dataset.py', '-i', in_dataset, '-o', out_folder, '-n', '1',
                            '--report', '-f')

    assert ret.success
