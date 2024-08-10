import argparse
from src.config import Config
from src.converter import JetClassConverter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert JetClass dataset from .root to .h5")
    parser.add_argument("--n_signal_files_in_train", type=int, default=5, help="Number of signal files in train set")
    parser.add_argument("--n_signal_files_in_test", type=int, default=2, help="Number of signal files in test set")
    parser.add_argument("--n_signal_files_in_valid", type=int, default=2, help="Number of signal files in validation set")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--label_signal", type=int, default=1, help="Label for signal events")
    parser.add_argument("--label_bkg", type=int, default=0, help="Label for background events")
    parser.add_argument("--num_particles", type=int, default=128, help="Maximum number of particles per event")
    parser.add_argument("--path", type=str, default="/raven/u/phebbar/Work/PELICAN_Btag/b_datasets", help="Path to the JetClass dataset")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = Config(args)
    converter = JetClassConverter(config)
    converter.convert()

if __name__ == "__main__":
    main()