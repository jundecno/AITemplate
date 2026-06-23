import os

import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))


def get_dataset_path():
    return root_path + "/datasets/"


def get_tools_path():
    return root_path + "/tools/"


def get_configs_path():
    return root_path + "/configs/"


def get_info_path():
    return get_dataset_path() + "/_info/"


def get_statis_path():
    return get_dataset_path() + "/_info/statis/"


def get_labels_path():
    return get_dataset_path() + "/_info/labels/"


def get_feature_path():
    return get_dataset_path() + "/_feat/"


def get_files(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            yield file


def get_file_paths(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            yield os.path.join(root, file)
