# Dump preprocessed data in --data_dir.

import yaml
import argparse
from preprocess.preprocessor import preprocess_vft_data


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configs/preprocess_dataset.yml",help="Config file with dataset parameters")
parser.add_argument("-r", "--data_root", default="data/VFT/",help="Root directory with data")
parser.add_argument("-d", "--data_dir", default="data/VFT/preprocessed",help="Directory to extract data")
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


preprocess_vft_data(args.data_root, args.data_dir)