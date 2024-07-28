import uproot
import awkward as ak
import vector 
import os
import glob
import h5py
import numpy as np

def main():
    
#Change initial hyper-paramters, the file-path to the required object and "label" definition
    n_signal_files_in_train = 5  #10 in each train** folder, each file has 100k events
    n_signal_files_in_test = 2
    n_signal_files_in_valid = 2
    shuffle                 = False
    label_signal=1 
    label_bkg   =0 #'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 
    #'label_H4q','label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'
    num_particles=128
    path = "/raven/u/phebbar/Work/PELICAN_Btag/b_datasets"  #path to jetclass dataset 
    
    #Dictionaries
    #dict_labels= {0:'label_QCD', 1:'label_Hbb', 2:'label_Hcc', 3:'label_Hgg', 4:'label_H4q',5:'label_Hqql', 6:'label_Zqq', 7:'label_Wqq', 8:'label_Tbqq', 9:'label_Tbl'}
    #dict_add = {0:'part_deta',1:'part_dphi',2:'part_d0val',3:'part_d0err',4:'part_dzval',5:'part_dzerr',6:'part_charge',7:'part_isChargedHadron',8:'part_isNeutralHadron',9:'part_isPhoton',10:'part_isElectron',11:'part_isMuon'}
    dict_files = {0:"ZJetsToNuNu*",1:"HToBB*",2:"HToCC*",3:"HToGG*",4:"HToWW4Q*",5:"HToWW2Q1L*",6:"ZToQQ*",7:"WToQQ*",8:"TTBar*",9:"TTBarLep*"}
    
    #Zero padding function taken from Particle Transformer (Huilin Qu) git repo
    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    #Function to convert .root to return numpy arrays
    def root_to_numpy(filepath):
        #Features list 
        particle_features_pmu=['part_energy','part_px', 'part_py', 'part_pz' ]
        particle_features_cyl=['part_energy','part_pt', 'part_eta', 'part_phi']
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q','label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
        addition=['part_deta','part_dphi','part_d0val','part_d0err','part_dzval','part_dzerr','part_charge','part_isChargedHadron','part_isNeutralHadron','part_isPhoton','part_isElectron','part_isMuon']
        # [0-9] labels , [0-11] addition 
        
        #Opening root
        table = uproot.open(filepath)['tree'].arrays()
        p4 = vector.zip({'energy': table['part_energy'],'px': table['part_px'], 'py': table['part_py'], 'pz': table['part_pz']})
        table['part_pt'] = p4.pt
        table['part_eta'] = p4.eta
        table['part_phi'] = p4.phi

        #Hyperparameters
        max_num_particles=num_particles

        #Conversion
        pmu = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features_pmu], axis=-1)
        cyl = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features_cyl], axis=-1)
        jet = np.stack([ak.to_numpy(table[n]) for n in jet_features], axis=-1)
        y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=-1)
        add = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in addition], axis=-1)

        #Logging
        print("Converting ", filepath.split("/" )[-1]," to numpy arrays")

        #return
        return pmu, cyl, jet, y ,add
    
        

    #File Path defining   
    
    train_Higgs_set = glob.glob(os.path.join(path,'train*',dict_files[label_signal]))   
    train_Background_set = glob.glob(os.path.join(path,'train*',dict_files[label_bkg]))

    test_Higgs_set = glob.glob(os.path.join(path,'test*',dict_files[label_signal]))  
    test_Background_set = glob.glob(os.path.join(path,'test*',dict_files[label_bkg]))

    val_Higgs_set = glob.glob(os.path.join(path,'val*',dict_files[label_signal]))  
    val_Background_set = glob.glob(os.path.join(path,'val*',dict_files[label_bkg]))

    def concat(glob_set_Higgs, glob_set_Background, number_of_signal_files): 
        
        _PMU, _CYL, _JET, _LABEL, _ADD = root_to_numpy(glob_set_Higgs[-1]) #First loads in the last training file for np.concatenate to work in for loop
        dummy_events = _PMU.shape[0]

        for i in range(number_of_signal_files):  #Concatenates the signal samples of Higgs
            _pmu, _cyl, _jet, _y, _add = root_to_numpy(glob_set_Higgs[i])
            _PMU = np.concatenate((_PMU,_pmu), axis=0)
            _CYL = np.concatenate((_CYL,_cyl), axis=0)
            _JET = np.concatenate((_JET,_jet), axis=0)
            _LABEL = np.concatenate((_LABEL,_y), axis=0)
            _ADD = np.concatenate((_ADD,_add), axis=0)
            
            assert _PMU.shape[1:]==_pmu.shape[1:], "Error in file conversion loop"
            assert _CYL.shape[1:]==_cyl.shape[1:], "Error in file conversion loop"
            assert _JET.shape[1:]==_jet.shape[1:], "Error in file conversion loop"
            assert _LABEL.shape[1:]==_y.shape[1:], "Error in file conversion loop"
            assert _ADD.shape[1:]==_add.shape[1:], "Error in file conversion loop"
            
        for i in range(number_of_signal_files): #Concatenates background ontop of signal sample
            _pmu, _cyl, _jet, _y ,_add = root_to_numpy(glob_set_Background[i])
            _PMU = np.concatenate((_PMU,_pmu), axis=0)
            _CYL = np.concatenate((_CYL,_cyl), axis=0)
            _JET = np.concatenate((_JET,_jet), axis=0)
            _LABEL = np.concatenate((_LABEL,_y), axis=0)
            _ADD = np.concatenate((_ADD,_add), axis=0)
            assert _PMU.shape[1:]==_pmu.shape[1:], "Error in file conversion loop"
            assert _CYL.shape[1:]==_cyl.shape[1:], "Error in file conversion loop"
            assert _JET.shape[1:]==_jet.shape[1:], "Error in file conversion loop"
            assert _LABEL.shape[1:]==_y.shape[1:], "Error in file conversion loop"
            assert _ADD.shape[1:]==_add.shape[1:], "Error in file conversion loop"

        _PMU = _PMU[dummy_events:] 
        _CYL =  _CYL[dummy_events:]
        _JET = _JET[dummy_events:]
        _LABEL = _LABEL[dummy_events:]
        _ADD = _ADD[dummy_events:]

        return _PMU, _CYL, _JET, _LABEL, _ADD


    train_PMU, train_CYL, train_JET , train_LABEL, train_ADD = concat(train_Higgs_set, train_Background_set, n_signal_files_in_train)
    test_PMU, test_CYL, test_JET , test_LABEL, test_ADD = concat(test_Higgs_set, test_Background_set, n_signal_files_in_test)
    val_PMU, val_CYL, val_JET , val_LABEL , val_ADD = concat(val_Higgs_set, val_Background_set, n_signal_files_in_valid)

    def unison_shuffled_copies(a, b , c , d, e):  #Random shuffler not necessary? PELICAN shuffles
        assert len(a) == len(b)
        assert len(b) == len(c)
        assert len(c) == len(d)
        assert len(d) == len(e)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p], e[p]
    
    if shuffle==True:
        train_PMU, train_CYL, train_JET , train_LABEL, train_ADD= unison_shuffled_copies(train_PMU, train_CYL, train_JET , train_LABEL)
        test_PMU, test_CYL, test_JET , test_LABEL, test_ADD= unison_shuffled_copies(test_PMU, test_CYL, test_JET , test_LABEL)
        val_PMU, val_CYL, val_JET , val_LABEL, val_ADD= unison_shuffled_copies(val_PMU, val_CYL, val_JET , val_LABEL)

    #Creating .h5 files 
    print("Creating train, test, valid files")
    hf1 = h5py.File('train.h5', 'w')
    hf2 = h5py.File('test.h5', 'w')
    hf3 = h5py.File('valid.h5', 'w')

    hf1.create_dataset('Pmu', data=train_PMU)
    hf2.create_dataset('Pmu', data=test_PMU)
    hf3.create_dataset('Pmu', data=val_PMU)
    
    hf1.create_dataset('Cyl', data=train_CYL)
    hf2.create_dataset('Cyl', data=test_CYL)
    hf3.create_dataset('Cyl', data=val_CYL)

    hf1.create_dataset('Jet', data=train_JET)
    hf2.create_dataset('Jet', data=test_JET)
    hf3.create_dataset('Jet', data=val_JET)

    hf1.create_dataset('is_signal', data=train_LABEL[:,1].astype('int'))  #tag index 1 is HtoBB and 2 is HtoCC
    hf2.create_dataset('is_signal', data=test_LABEL[:,1].astype('int')) #tag index 6 is Ztoqqbar
    hf3.create_dataset('is_signal', data=val_LABEL[:,1].astype('int'))
    
    hf1.create_dataset('scalars', data=train_ADD)  
    hf2.create_dataset('scalars', data=test_ADD) #index [10-21] is additional info
    hf3.create_dataset('scalars', data=val_ADD)

    #Number of non-zero events
    Pmu_norm1 = np.linalg.norm(train_PMU, axis=-1)
    Pmu_norm2 = np.linalg.norm(test_PMU, axis=-1)
    Pmu_norm3 = np.linalg.norm(val_PMU, axis=-1)
    
    Nobj1 = np.count_nonzero(Pmu_norm1, axis=-1)
    Nobj2 = np.count_nonzero(Pmu_norm2, axis=-1)
    Nobj3 = np.count_nonzero(Pmu_norm3, axis=-1)

    
    hf1.create_dataset('Nobj', data=Nobj1)
    hf2.create_dataset('Nobj', data=Nobj2)
    hf3.create_dataset('Nobj', data=Nobj3)

    print("Conversion successful!")
  
    hf1.close()
    hf2.close()
    hf3.close()
    
if __name__=='__main__':
    main()
