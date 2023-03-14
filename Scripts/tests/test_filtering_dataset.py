#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, download_file_from_GDrive


#download_data(download_file_from_GDrive(), keys=['data.zip'])
tmp_dir = tempfile.TemporaryFile()


def test_help(script_runner):
    ret = script_runner.run('CCPM_filtering_dataset.py',
                            '--help')

    assert ret.success
