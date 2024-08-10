import numpy as np
import uproot
import awkward as ak
import vector

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

def root_to_numpy(filepath, num_particles):
    particle_features_pmu = ['part_energy', 'part_px', 'part_py', 'part_pz']
    particle_features_cyl = ['part_energy', 'part_pt', 'part_eta', 'part_phi']
    jet_features = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
    labels = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    addition = ['part_deta', 'part_dphi', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr', 'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron', 'part_isMuon']

    table = uproot.open(filepath)['tree'].arrays()
    p4 = vector.zip({'energy': table['part_energy'], 'px': table['part_px'], 'py': table['part_py'], 'pz': table['part_pz']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    pmu = np.stack([ak.to_numpy(_pad(table[n], maxlen=num_particles)) for n in particle_features_pmu], axis=-1)
    cyl = np.stack([ak.to_numpy(_pad(table[n], maxlen=num_particles)) for n in particle_features_cyl], axis=-1)
    jet = np.stack([ak.to_numpy(table[n]) for n in jet_features], axis=-1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=-1)
    add = np.stack([ak.to_numpy(_pad(table[n], maxlen=num_particles)) for n in addition], axis=-1)

    print(f"Converting {filepath.split('/')[-1]} to numpy arrays")

    return pmu, cyl, jet, y, add

def unison_shuffled_copies(*arrays):
    assert len(set(len(arr) for arr in arrays)) == 1, "All arrays must have the same length"
    p = np.random.permutation(len(arrays[0]))
    return tuple(arr[p] for arr in arrays)