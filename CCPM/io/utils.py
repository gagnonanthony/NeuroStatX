import inspect
import os
import shutil

import pandas as pd
from fpdf import FPDF

"""
Some function comes from the scilpy toolbox. Please see : 
https://github.com/scilus/scilpy
"""

def load_df_in_any_format(file):
    """
    Load dataset in any .csv or .xlsx format.
    :param df:
    :return:
    """
    _, ext = os.path.splitext(file)
    if ext == '.csv':
        df = pd.read_csv(file)
    if ext == '.xlsx':
        df = pd.read_excel(file)

    return df


def add_overwrite_arg(p):
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of existing output files.')


def add_verbose_arg(p):
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='If true, produce verbose output.')


def assert_input(p, required, optional=None):
    """
    Function to validate the existence of an input file.
    From the scilpy toolbox : https://github.com/scilus/scilpy
    :param p:           Parser
    :param required:    Paths to assert the existence
    :param optional:    Paths to assert optional arguments
    :return:
    """
    def check(path):
        if not os.path.isfile(path):
            p.error('File {} does not exist.'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for file in required:
        check(file)
    for file in optional or []:
        if file is not None:
            check(file)


def assert_output(p, args, required, optional=None, check_dir=True):
    """
    Validate that output exist and force the use of -f.
    From the scilpy toolbox : https://github.com/scilus/scilpy
    :param p:           Parser.
    :param args:        Arguments.
    :param required:    Required paths to assert.
    :param optional:    Optional paths to assert.
    :param check_dir:   Validate if output directory exist.
    :return:
    """
    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            p.error('Output file {} exists. Select the -f to force '
                    'overwriting of the existing file.'.format(path))

        if check_dir:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                p.error('Directory {} is not created for the output file.'.format(path_dir))

    if isinstance(required, str):
        required = [required]
    if isinstance(optional, str):
        optional = [optional]

    for file in required:
        check(file)
    for file in optional or []:
        if file:
            check(file)


def assert_output_dir_exist(p, args, required, optional=None, create_dir=True):
    """
    Validate the existence of the output directory.
    From the scilpy toolbox : https://github.com/scilus/scilpy
    :param p:               Parser.
    :param args:            Arguments.
    :param required:        Required paths to validate.
    :param optional:        Optional paths to validate.
    :param create_dir:      Option to create the directory if it does not already exist.
    :return:
    """
    def check(path):
        if not os.path.isdir(path):
            if not create_dir:
                p.error('Output directory {} does not exist. Use create_dir = True.'.format(path))
            else:
                os.makedirs(path, exist_ok=True)
        if os.listdir(path):
            if not args.overwrite:
                p.error('Output directory {} is not empty. Use -f to overwrite the existing '
                        'content.'.format(path))
            else:
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

    if isinstance(required, str):
        required = [required]
    if isinstance(optional, str):
        optional = [optional]

    for cur_dir in required:
        check(cur_dir)
    for opt_dir in optional or []:
        if opt_dir:
            check(opt_dir)


def get_data_dir():
    """
    Return a data directory within the CCPM repository
    :return:
    data_dir: data path
    """
    import CCPM
    module_path = inspect.getfile(CCPM)

    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(module_path)) + "/data/")

    return data_dir


class PDF(FPDF):
    """
    Class object to initialize reports to output recommendation in pdf
    format.
    """
    def header(self):
        path = os.path.join(get_data_dir()+'CCPM.png')
        self.image(path, 10, 8, 33)
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'{num} : {label}', 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, name):
        with open(name, 'rb') as fh:
            txt = fh.read().decode('latin-1')
        self.set_font('Times', '', 12)
        self.multi_cell(0, 5, txt)
        self.ln()

    def print_chapter(self, num, title, name):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name)
