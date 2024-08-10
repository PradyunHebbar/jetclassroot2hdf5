# JetClass Dataset Converter (.root to .h5)

This project contains Python scripts for converting the JetClass dataset from .root format to .h5 format, making it suitable for neural network model training.

## Description

The JetClass dataset is a large-scale dataset for deep learning in jet physics ([Zenodo link](https://zenodo.org/records/6619768)). This script allows users to convert the original .root files into .h5 files, with options to select specific signal events, specify the number of files to process, and control the maximum number of particles per event.

## Features

- Convert JetClass dataset from .root to .h5 format
- Specify the number of signal files for train, test, and validation sets
- Select specific signal events (H, W, Z boson decays, top quark decays)
- Control the maximum number of particles per event
- Option to shuffle the data
- Zero-padding for events with fewer particles
- Multiprocessing support for improved performance

## Requirements

- uproot==5.2.2
- awkward==2.6.1
- vector
- h5py
- numpy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/jetclassroot2hdf5.git
   cd jetclassroot2hdf5
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script using the following command:

```
python main.py [arguments]
```

Available arguments:

- `--n_signal_files_in_train`: Number of signal files to use for training (default: 5)
- `--n_signal_files_in_test`: Number of signal files to use for testing (default: 2)
- `--n_signal_files_in_valid`: Number of signal files to use for validation (default: 2)
- `--shuffle`: Whether to shuffle the dataset. Use 'true' or 'false' (default: 'false')
- `--label_signal`: Select the signal event type (1-9) (default: 1)
- `--label_bkg`: Select the background event type (0-9) (default: 0)
- `--num_particles`: Maximum number of particles per event (default: 128)
- `--path`: Path to the JetClass dataset (default: "/raven/u/phebbar/Work/PELICAN_Btag/b_datasets")

Example usage:

```
python main.py --n_signal_files_in_train 10 --n_signal_files_in_test 3 --n_signal_files_in_valid 3 --shuffle true --label_signal 2 --num_particles 256
```

This command will run the conversion process with 10 signal files for training, 3 for testing, 3 for validation, shuffle the data, use label 2 for signal events, and set the maximum number of particles to 256.

## Output

The script will generate three .h5 files: `train.h5`, `test.h5`, and `valid.h5`. Each file contains the following datasets:

1. `Pmu`: 4-momentum of particles per event (E, px, py, pz)
2. `Cyl`: Cylindrical coordinates of particles per event (Energy, pt, eta, phi)
3. `is_signal`: Binary value indicating whether the event is the selected signal or not
4. `scalars`: Additional information per particle as provided by JetClass
5. `Nobj`: Number of particles per event
6. `Jet`: Jet 4-momentum

## Dataset Information

The JetClass dataset contains 9 different signal events (top quark, H, W, Z boson decays) and 1 background event (QCD jets). Each file in the dataset contains 100,000 events.

For more information about the dataset, please refer to the [arXiv paper](https://arxiv.org/pdf/2202.03772) and the [Zenodo dataset page](https://zenodo.org/records/6619768).

## Citation

If you use the JetClass dataset in your research, please cite:

```
@dataset{JetClass,
  author       = "Qu, Huilin and Li, Congqiao and Qian, Sitian",
  title        = "{JetClass}: A Large-Scale Dataset for Deep Learning in Jet Physics",
  month        = "jun",
  year         = "2022",
  publisher    = "Zenodo",
  version      = "1.0.0",
  doi          = "10.5281/zenodo.6619768",
  url          = "https://doi.org/10.5281/zenodo.6619768"
}
```

## Contact

For any questions or issues, please open an issue on this GitHub repository or contact:

Pradyun Hebbar
pradyun.hebbar@gmail.com
