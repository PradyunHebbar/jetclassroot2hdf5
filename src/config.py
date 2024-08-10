class Config:
    def __init__(self, args):
        self.n_signal_files_in_train = args.n_signal_files_in_train
        self.n_signal_files_in_test = args.n_signal_files_in_test
        self.n_signal_files_in_valid = args.n_signal_files_in_valid
        self.shuffle = args.shuffle
        self.label_signal = args.label_signal
        self.label_bkg = args.label_bkg
        self.num_particles = args.num_particles
        self.path = args.path

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())