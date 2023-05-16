import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=['testing_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run('CCPM_factor_analysis.py',
                            '--help')

    assert ret.success

# TODO: adapt test function.
def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dataset = os.path.join(get_home(), 'data/factor_data.xlsx')
    out_folder = os.path.join(get_home(), 'data/factor_results/')
    ret = script_runner.run('CCPM_factor_analysis.py', '-i', in_dataset, '-o', out_folder,
                            '-s', 'subjectkey', '-f')

    assert ret.success
