import argparse
import os
import num2words
from . import centipede_generator
import os.path as osp
import sys
import datetime


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


_this_dir = osp.dirname(__file__)
running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime("%Y_%m_%d-%X"))

_base_dir = osp.join(_this_dir, "..")
add_path(_base_dir)


def bypass_frost_warning():
    return 0


def get_base_dir():
    return _base_dir


def get_time():
    return time


def get_abs_base_dir():
    return osp.abspath(_base_dir)


TASK_DICT = {
    "Centipede": [3, 5, 7] + [4, 6, 8, 10, 12, 14, 18] + [20, 24, 30, 40, 50],
    "CentipedeTT": [6],
    "CpCentipede": [3, 5, 7] + [4, 6, 8, 10, 12, 14],
    "Reacher": [0, 1, 2, 3, 4, 5, 6, 7],
    "Snake": list(range(3, 10)) + [10, 20, 40],
}
OUTPUT_BASE_DIR = os.path.join(get_abs_base_dir(), "environments", "assets")


def save_xml_files(model_names, xml_number, xml_contents):
    # get the xml path ready
    number_str = num2words.num2words(xml_number)
    xml_names = model_names + number_str[0].upper() + number_str[1:] + ".xml"
    xml_file_path = os.path.join(OUTPUT_BASE_DIR, xml_names)

    # save the xml file
    f = open(xml_file_path, "w")
    f.write(xml_contents)
    f.close()


GENERATOR_DICT = {
    "Centipede": centipede_generator.generate_centipede,
    "CentipedeTT": centipede_generator.generate_centipede,
}

if __name__ == "__main__":
    # parse the parameters
    parser = argparse.ArgumentParser(description="xml_asset_generator.")
    parser.add_argument("--env_name", type=str, default="Centipede")
    args = parser.parse_args()

    # generator the environment xmls
    for i_leg_num in TASK_DICT[args.env_name]:
        xml_contents = GENERATOR_DICT[args.env_name](i_leg_num)
        save_xml_files(args.env_name, i_leg_num, xml_contents)
