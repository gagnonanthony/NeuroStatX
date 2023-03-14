# -*- coding: utf-8 -*-
import os
import logging
import zipfile
import pathlib

import gdown


def save_files_dict():
    """ Getting dictionary file list from GDrive. """
    return {'data.zip':
            '1WNytflyfZ9gCge1hCCIkDtYnSGpbUsnY'}


def download_file_from_GDrive(id, destination):
    """
    Function to download a zip file from a Google Drive.
    :param id:
    :param destination:
    :return:
    """
    gdown.download(id=id, output=destination, quiet=True)


def get_home():
    """ Set home directory of CCPM for testing. """
    if 'CCPM_HOME' in os.environ:
        ccpm_home = os.environ['CCPM_HOME']
    else:
        ccpm_home = os.path.join(os.path.expanduser('~'), '.ccpm')
    return ccpm_home


def download_data(files_dict, keys=None):
    """
    Function to download and extract data with gdown.
    Using structure from https://github.com/scilus/scilpy/blob/master/scilpy/io/fetcher.py
    :param files_dict:
    :param keys:
    :return:
    """

    ccpm_home = get_home()

    if not os.path.exists(ccpm_home):
        os.makedirs(ccpm_home)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        id = files_dict[f]
        full_path = os.path.join(ccpm_home, f)
        full_path_no_ext, ext = os.path.splitext(full_path)

        if not os.path.isdir(full_path_no_ext):
            if ext == '.zip' and not os.path.isdir(full_path_no_ext):
                logging.warning('Downloading file and extracting {} from id {} to {}'.format(
                    f, id, ccpm_home
                ))

                download_file_from_GDrive(id, full_path)

                try:
                    z = zipfile.ZipFile(full_path)
                    zipinfos = z.infolist()
                    root_dir = pathlib.Path(
                        zipinfos[0].filename).parts[0] + '/'
                    assert all([s.startswith(root_dir) for s in z.namelist()])
                    with zipfile.ZipFile(full_path, 'r') as zip_ref:
                        zip_ref.extractall(ccpm_home)

                except AssertionError:
                    z.extractall(full_path)
            else:
                raise NotImplementedError('Function was expected a zip file.')

        else:
            logging.warning('Data already downloaded.')
