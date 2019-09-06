import argparse
import os
import pickle

import yaml

from data_loader import twitter_airline_dataset
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import MosesEncoder
from training import train_manager


def save_data(task, file_name = 'task.pkl'):
    if not os.path.exists('.preprocess'):
        os.makedirs('.preprocess')
    with open('.preprocess/' + file_name, 'wb') as filehandler:
        pickle.dump(task, filehandler)

def main():
    arg_parser = argparse.ArgumentParser("Minimalist Transformer")
    arg_parser.add_argument("mode", choices=["preprocess", "train"],
                    help="train a model or test or translate")
    arg_parser.add_argument('-f', '--config', default='default.yaml', 
                    help='Configuration file to load.')
    args = arg_parser.parse_args()
    configs = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

    if args.mode == "train":
        train_manager(configs)

    elif args.mode == "preprocess":
        train, _, test =  twitter_airline_dataset()
        text_encoder = MosesEncoder(train['source'])
        label_encoder = LabelEncoder(train['target'])
        with open('.preprocess.pkl', 'wb') as filehandler:
            pickle.dump((text_encoder, label_encoder, train, test), filehandler)

    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
