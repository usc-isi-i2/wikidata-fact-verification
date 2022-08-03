import os
import shutil


def prepare_run_files_directory(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

    print(f'Storing run files at: {filepath}')
    os.makedirs(filepath)


def delete_dir(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
