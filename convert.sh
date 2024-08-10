#!/bin/bash

## With shuffle 
#python main.py --n_signal_files_in_train 2 --n_signal_files_in_test 1 --n_signal_files_in_valid 1 --shuffle --label_signal 1 --label_bkg 0 --num_particles 128

python main.py --n_signal_files_in_train 2 --n_signal_files_in_test 1 --n_signal_files_in_valid 1 --label_signal 1 --label_bkg 0 --num_particles 128