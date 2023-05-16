import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run('CCPM_creating_medical_statistics.py',
                            '--help')

    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/demographics.xlsx')
    out_table = os.path.join(get_home(), 'data/table.xlsx')
    ret = script_runner.run('CCPM_creating_medical_statistics.py', '-i', in_dataset, '-o', out_table,
                            '--identifier_column', 'subid', '--total_variables', 'Sex', 'Age', 'IQ',
                            '--categorical_variables', 'Sex', '--var_names', 'Sex', 'Age', 'Quotient',
                            '--apply_yes_or_no', '-f')

    assert ret.success
