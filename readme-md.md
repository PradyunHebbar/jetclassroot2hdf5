# JetClass Dataset Converter

This project contains a Python script for converting the JetClass dataset from .root format to .h5 format, making it suitable for neural network model training.

## Description

The JetClass dataset is a large-scale dataset for deep learning in jet physics. This script allows users to convert the original .root files into .h5 files, with options to select specific signal events, specify the number of files to process, and control the maximum number of particles per event.

## Features

- Convert JetClass dataset from .root to .h5 format
- Specify the number of signal files for train, test, and validation sets
- Select specific signal events (H, W, Z boson decays, top quark decays)
- Control the maximum number of particles per event
- Option to shuffle the data
- Zero-padding for events with fewer particles

## Requirements

- Python 3.x
- uproot
- awkward
- vector
- h5py
- numpy

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/jetclass-converter.git
   cd jetclass-converter
   ```

2. Install the required packages:
   ```
   pip install uproot awkward vector h5py numpy
   ```

3. Modify the following parameters in the `main()` function as needed:
   - `n_signal_files_in_train`: Number of signal files to use for training
   - `n_signal_files_in_test`: Number of signal files to use for testing
   - `n_signal_files_in_valid`: Number of signal files to use for validation
   - `shuffle`: Set to True if you want to shuffle the data
   - `label_signal`: Select the signal event type (1-9)
   - `num_particles`: Maximum number of particles per event
   - `path`: Path to the JetClass dataset

4. Run the script:
   ```
   python jetclass_converter.py
   ```

5. The script will generate three .h5 files: `train.h5`, `test.h5`, and `valid.h5`.

## Output Format

The generated .h5 files contain the following datasets:

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

If you use this code or the JetClass dataset in your research, please cite:

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

## License

[Include your chosen license here]

## Contributing

[Include guidelines for contributing to your project, if applicable]

## Contact

[Your Name or GitHub username]
[Your email or other contact information]
