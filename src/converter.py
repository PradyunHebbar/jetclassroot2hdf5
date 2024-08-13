import os
import glob
import numpy as np
import h5py
import multiprocessing as mp
from functools import partial
from src.utils import root_to_numpy, unison_shuffled_copies

class JetClassConverter:
    def __init__(self, config):
        self.config = config
        self.dict_files = {0:"ZJetsToNuNu*", 1:"HToBB*", 2:"HToCC*", 3:"HToGG*", 
                           4:"HToWW4Q*", 5:"HToWW2Q1L*", 6:"ZToQQ*", 7:"WToQQ*", 
                           8:"TTBar*", 9:"TTBarLep*"}

    def get_file_sets(self):
        train_Higgs_set = glob.glob(os.path.join(self.config.path,  
                                    self.dict_files[self.config.label_signal]))
        train_Background_set = glob.glob(os.path.join(self.config.path,  
                                         self.dict_files[self.config.label_bkg]))
        test_Higgs_set = glob.glob(os.path.join(self.config.path, 'test*', 
                                   self.dict_files[self.config.label_signal]))
        test_Background_set = glob.glob(os.path.join(self.config.path, 'test*', 
                                        self.dict_files[self.config.label_bkg]))
        val_Higgs_set = glob.glob(os.path.join(self.config.path, 'val*', 
                                  self.dict_files[self.config.label_signal]))
        val_Background_set = glob.glob(os.path.join(self.config.path, 'val*', 
                                       self.dict_files[self.config.label_bkg]))
        return (train_Higgs_set, train_Background_set, test_Higgs_set, 
                test_Background_set, val_Higgs_set, val_Background_set)

    def process_file_set(self, file_set, num_files):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(partial(root_to_numpy, num_particles=self.config.num_particles), 
                               file_set[:num_files])
        return tuple(np.concatenate(data) for data in zip(*results))

    def convert(self):
        file_sets = self.get_file_sets()
        
        train_signal = self.process_file_set(file_sets[0], self.config.n_signal_files_in_train)
        train_background = self.process_file_set(file_sets[1], self.config.n_signal_files_in_train)
        train_data = tuple(np.concatenate((s, b)) for s, b in zip(train_signal, train_background))

        test_signal = self.process_file_set(file_sets[2], self.config.n_signal_files_in_test)
        test_background = self.process_file_set(file_sets[3], self.config.n_signal_files_in_test)
        test_data = tuple(np.concatenate((s, b)) for s, b in zip(test_signal, test_background))

        val_signal = self.process_file_set(file_sets[4], self.config.n_signal_files_in_valid)
        val_background = self.process_file_set(file_sets[5], self.config.n_signal_files_in_valid)
        val_data = tuple(np.concatenate((s, b)) for s, b in zip(val_signal, val_background))

        if self.config.shuffle:
            train_data = unison_shuffled_copies(*train_data)
            test_data = unison_shuffled_copies(*test_data)
            val_data = unison_shuffled_copies(*val_data)

        self.create_h5_file('train.h5', *train_data)
        self.create_h5_file('test.h5', *test_data)
        self.create_h5_file('valid.h5', *val_data)

    def create_h5_file(self, filename, PMU, CYL, JET, LABEL, ADD):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('Pmu', data=PMU, chunks=True)
            hf.create_dataset('Cyl', data=CYL, chunks=True)
            hf.create_dataset('Jet', data=JET, chunks=True)
            hf.create_dataset('is_signal', data=LABEL[:, self.config.label_signal].astype('int'), chunks=True)
            hf.create_dataset('scalars', data=ADD, chunks=True)
            
            Pmu_norm = np.linalg.norm(PMU, axis=-1)
            Nobj = np.count_nonzero(Pmu_norm, axis=-1)
            hf.create_dataset('Nobj', data=Nobj, chunks=True)

        print(f"Created {filename}")