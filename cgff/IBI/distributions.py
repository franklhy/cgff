from glob import glob
import numpy as np
from scipy import interpolate
from scipy import signal
from plato import data, neigh

class distributions:
    def __init__(self):
        pass

    def _sort_pairs(self, pairs):
        pairs = np.array(pairs)
        for i in range(len(pairs)):
            if pairs[i,0] > pairs[i,1]:
                tmp = pairs[i,0]
                pairs[i,0] = pairs[i,1]
                pairs[i,1] = tmp
        ### sort pairs by first column, then the second column
        pairs = pairs[pairs[:,1].argsort()]                    ### first sort does not need to be stable
        pairs = pairs[pairs[:,0].argsort(kind='mergesort')]    ### stable sort
        return pairs

    def _bond_length_distribution(self, cutofflo, cutoffhi, nbins, bonds, atom_position, idmapping_atom):
        bond_vector = atom_position[ idmapping_atom[bonds[:,2]] ] - atom_position[ idmapping_atom[bonds[:,3]] ]
        bond_length = (np.sum(bond_vector**2, axis=1))**0.5
        bins = np.linspace(cutofflo, cutoffhi, nbins+1)
        vols = 4.0 / 3.0 * np.pi * (bins[1:]**3 - bins[:-1]**3)
        counts, _ = np.histogram(bond_length, bins)
        rou = counts / vols / len(bonds)
        bins_center = (bins[1:] + bins[:-1]) * 0.5
        
        norm = np.sum(rou) * (bins_center[1] - bins_center[0])
        rou /= norm
        return np.column_stack((bins_center, rou))

    def _angle_distribution(self, nbins, angles, atom_position, idmapping_atom):
        bond_vector1 = atom_position[ idmapping_atom[angles[:,2]] ] - atom_position[ idmapping_atom[angles[:,3]] ]
        bond_vector2 = atom_position[ idmapping_atom[angles[:,4]] ] - atom_position[ idmapping_atom[angles[:,3]] ]
        bond_length1 = (np.sum(bond_vector1**2, axis=1))**0.5
        bond_length2 = (np.sum(bond_vector2**2, axis=1))**0.5
        costheta = np.sum(bond_vector1 * bond_vector2, axis=1) / bond_length1 / bond_length2 
        costheta[costheta < -1] = -1
        costheta[costheta > 1] = 1
        theta = np.arccos(costheta)    ### rad
        theta = theta / np.pi * 180    ### degree
        bins = np.linspace(0,180,nbins)    ### remove one bin for histogram, bacause it will be added later for 180 degree
        areas = 2 * np.pi * ( np.cos(bins[:-1]/180*np.pi) - np.cos(bins[1:]/180*np.pi) )    ### area of a spherical segment = 2 * pi * R * h, 
                                                                                            ### here R = 1, h = np.cos(bins[:-1]/180*np.pi) - np.cos(bins[1:]/180*np.pi)
        counts, _ = np.histogram(theta, bins)
        rou = counts / areas / len(angles)
        bins = bins[:-1]
        ### Create distribution for 180 degree. It is required by LAMMPS that the angle potential table cover both 0 degree and 180 degree
        bins = np.append(bins, 180)
        rou = np.append(rou, rou[-1])

        norm = np.sum(rou) * (bins[1] - bins[0])
        rou /= norm
        return np.column_stack((bins, rou))

    def _dihedral_distribution(self, nbins, dihedrals, atom_position, idmapping_atom):
        bond_vector1 = atom_position[ idmapping_atom[dihedrals[:,3]] ] - atom_position[ idmapping_atom[dihedrals[:,2]] ]
        bond_vector2 = atom_position[ idmapping_atom[dihedrals[:,4]] ] - atom_position[ idmapping_atom[dihedrals[:,3]] ]
        bond_vector3 = atom_position[ idmapping_atom[dihedrals[:,5]] ] - atom_position[ idmapping_atom[dihedrals[:,4]] ]
        plane_norm1 = np.cross(bond_vector1, bond_vector2)
        plane_norm2 = np.cross(bond_vector2, bond_vector3)
        plane_norm_length1 = (np.sum(plane_norm1**2, axis=1))**0.5
        plane_norm_length2 = (np.sum(plane_norm2**2, axis=1))**0.5
        bond_length2 = (np.sum(bond_vector2**2, axis=1))**0.5
        costheta = np.sum(plane_norm1 * plane_norm2, axis=1) / plane_norm_length1 / plane_norm_length2
        tmp = np.cross(plane_norm1, plane_norm2)
        sintheta = np.sum(tmp * bond_vector2, axis=1) / plane_norm_length1 / plane_norm_length2 / bond_length2
        theta = np.arctan2(sintheta, costheta)    ### rad
        theta = theta / np.pi * 180    ### degree
        bins = np.linspace(-180,180,nbins+1)
        counts, _ = np.histogram(theta, bins)
        rou = counts / len(dihedrals)
        bins_center = (bins[1:] + bins[:-1]) * 0.5

        norm = np.sum(rou) * (bins[1] - bins[0])
        rou /= norm
        return np.column_stack((bins_center, rou))

    def _improper_distribution(self, nbins, impropers, atom_position, idmapping_atom):
        bond_vector1 = atom_position[ idmapping_atom[impropers[:,3]] ] - atom_position[ idmapping_atom[impropers[:,2]] ]
        bond_vector2 = atom_position[ idmapping_atom[impropers[:,4]] ] - atom_position[ idmapping_atom[impropers[:,3]] ]
        bond_vector3 = atom_position[ idmapping_atom[impropers[:,5]] ] - atom_position[ idmapping_atom[impropers[:,4]] ]
        plane_norm1 = np.cross(bond_vector1, bond_vector2)
        plane_norm2 = np.cross(bond_vector2, bond_vector3)
        plane_norm_length1 = (np.sum(plane_norm1**2, axis=1))**0.5
        plane_norm_length2 = (np.sum(plane_norm2**2, axis=1))**0.5
        bond_length2 = (np.sum(bond_vector2**2, axis=1))**0.5
        costheta = np.sum(plane_norm1 * plane_norm2, axis=1) / plane_norm_length1 / plane_norm_length2
        tmp = np.cross(plane_norm1, plane_norm2)
        sintheta = np.sum(tmp * bond_vector2, axis=1) / plane_norm_length1 / plane_norm_length2 / bond_length2
        theta = np.arctan2(sintheta, costheta)    ### rad
        theta = theta / np.pi * 180    ### degree
        bins = np.linspace(-180,180,nbins+1)
        counts, _ = np.histogram(theta, bins)
        rou = counts / len(impropers)
        bins_center = (bins[1:] + bins[:-1]) * 0.5
        
        norm = np.sum(rou) * (bins_center[1] - bins_center[0])
        rou /= norm
        return np.column_stack((bins_center, rou))

    def read_distribution(self, path):
        """
        Read distributions of pairs, bonds, angles, dihedrals and inmpropers
        """
        rdf = {}
        bonddist = {}
        angledist = {}
        dihedraldist = {}
        improperdist = {}

        files = glob(path + "/rdf_*_*.txt")
        for f in files:
            key = f.split('/')[-1].split('.')[0].split('_')[1:]
            key = tuple([int(k) for k in key])
            rdf[key] = np.loadtxt(f)

        files = glob(path + "/bonddist_*.txt")
        for f in files:
            key = int(f.split('/')[-1].split('.')[0].split('_')[1])
            bonddist[key] = np.loadtxt(f)
        
        files = glob(path + "/angledist_*.txt")
        for f in files:
            key = int(f.split('/')[-1].split('.')[0].split('_')[1])
            angledist[key] = np.loadtxt(f)

        files = glob(path + "/dihedraldist_*.txt")
        for f in files:
            key = int(f.split('/')[-1].split('.')[0].split('_')[1])
            dihedraldist[key] = np.loadtxt(f)

        files = glob(path + "/improperdist_*.txt")
        for f in files:
            key = int(f.split('/')[-1].split('.')[0].split('_')[1])
            improperdist[key] = np.loadtxt(f)

        all_distributions = []
        all_distributions.append(rdf)
        all_distributions.append(bonddist)
        all_distributions.append(angledist)
        all_distributions.append(dihedraldist)
        all_distributions.append(improperdist)

        return all_distributions

    def rmsd(self, current_distribution, target_distribution, normed=False):
        """
        Calculate root-mean-squared deviation.
        
        normed: bool
            If ture, normalized the rmsd by the mean value of target distribution (non zero term)
        """
        np.testing.assert_array_almost_equal(current_distribution[:,0], target_distribution[:,0])
        P_current = current_distribution[:,1]
        P_target = target_distribution[:,1]
        both_not_zero_mask = (P_current != 0) * (P_target != 0)
        rmsd = np.sqrt(np.mean((P_current[both_not_zero_mask] - P_target[both_not_zero_mask])**2))
        if normed:
            rmsd /= np.mean(P_target[both_not_zero_mask])
        return rmsd

    def smooth_distribution(self, dist, s=0.02, w=5, p=3, mode="spline", minPratio=1e-3, periodic=False):
        """
        "mode" should be "spline", "savgol", "butt" or "none".
        When mode="spline", s (smoothing condition) is used by scipy.interpolate.splrep. w is ignored.
        When mode="savgol", w (the ratio of window length to data range) is used to calculate window length for scipy.signal.savgol_filter. s is ignored.
        """
        x = np.copy(dist[:,0])
        P = np.copy(dist[:,1])
        mask = P > np.max(P) * minPratio
        x_s = np.copy(x[mask])
        P_s = np.copy(P[mask])

        ### pre-smooth with Butterworth filter
        b, a = signal.butter(3, 0.1)
        filtered = signal.filtfilt(b, a, P_s, method='pad')
        mask_f = filtered < np.max(P) * minPratio
        filtered[mask_f] = np.max(P) * minPratio
        delta = ((P_s - filtered)/filtered)**2
        if np.mean(delta) > 0.001:    ### if the data is very noisy, then replace the original data with the filtered one
            P_s = filtered

        if mode == "spline":
            #weight = np.ones(len(P_s)) / np.mean(P_s)
            weight = np.ones(len(P_s)) / P_s
            if periodic:
                tck = interpolate.splrep(x_s, P_s, s=s**2*len(P_s), k=p, w=weight, per=1)
            else:
                tck = interpolate.splrep(x_s, P_s, s=s**2*len(P_s), k=p, w=weight)
            P_s = interpolate.splev(x_s, tck, der=0)
        elif mode == "savgol":
            if periodic:
                P_s = signal.savgol_filter(P_s, window_length=w, polyorder=p, mode="wrap")
            else:
                #P_s = signal.savgol_filter(P_s, window_length=w, polyorder=p, mode="interp")
                P_s = signal.savgol_filter(P_s, window_length=w, polyorder=p, mode="nearest")
        elif mode == "none":
            return np.column_stack((x,P))
        else:
            raise ValueError("Invalid smooth mode.")
        P[mask] = P_s
        minP = np.max(P) * minPratio
        P[P < minP] = minP
        return np.column_stack((x,P))

    def distributions(self, datafile, dumpfile, pair_cutoff_dict, pair_nbins_dict, bond_cutofflo_dict, bond_cutoffhi_dict, bond_nbins_dict, \
        angle_nbins=181, dihedral_nbins=360, improper_nbins=360, exclude_bond=True, exclude_angle=True, exclude_dihedral=True, exclude_improper=True, \
        inter_mol=True, intra_mol=True):
        """
        """
        d = data()
        d.read(datafile)
        d.set_traj_file(dumpfile)
        alltime = d.traj_time()
        d.load_traj(0)
        IX, IY, IZ = d.map("ix", "iy", "iz")
        if IX == -1 or IY == -1 or IZ == -1:
            print("Need image flag.")
            raise RuntimeError
        ne = neigh()

        _, _, _, _, atoms = d.snapshot()
        bonds,angles,dihedrals,impropers = d.topology()

        ### check pairs of atom types
        TYPE = d.map("type")
        alltypes = np.unique(atoms[:,TYPE])
        pairs = [[alltypes[i],alltypes[j]] for i in range(len(alltypes)) for j in range(i,len(alltypes))]
        tmpkeys1 = [[key[0], key[1]] for key in pair_cutoff_dict.keys()]
        tmpkeys2 = [[key[0], key[1]] for key in pair_nbins_dict.keys()]
        pairs = self._sort_pairs(pairs)
        tmpkeys1 = self._sort_pairs(tmpkeys1)
        tmpkeys2 = self._sort_pairs(tmpkeys2)
        if not np.array_equal(pairs, tmpkeys1) or not np.array_equal(pairs, tmpkeys2):
            print("Input pairs list is not correct.")
            raise RuntimeError

        ### check bond tpyes
        if len(bonds) != 0: 
            tmpkeys1 = list(bond_cutofflo_dict.keys())
            tmpkeys2 = list(bond_cutoffhi_dict.keys())
            tmpkeys3 = list(bond_nbins_dict.keys())
            bondtypes = np.unique(bonds[:,1])
            tmpkeys1.sort()
            tmpkeys2.sort()
            tmpkeys3.sort()
            if not np.array_equal(bondtypes, tmpkeys1) or not np.array_equal(bondtypes, tmpkeys2) or not np.array_equal(bondtypes, tmpkeys3):
                print("Keys in bond_cutofflo_dict, bond_cutoffhi_dict and/or bond_nbins_dict are not the same.")
                raise RuntimeError

        ### check flags
        if len(bonds) == 0 and exclude_bond:
            exclude_bond = False
        if len(angles) == 0 and exclude_angle:
            exclude_angle = False
        if len(dihedrals) == 0 and exclude_dihedral:
            exclude_dihedral = False
        if len(impropers) == 0 and exclude_improper:
            exclude_improper = False
        print(exclude_bond, exclude_angle, exclude_dihedral, exclude_improper)

        ### prepare
        ne.set_topology(d, inter_mol=inter_mol, intra_mol=intra_mol, exclude_bond=exclude_bond, exclude_angle=exclude_angle,\
                exclude_dihedral=exclude_dihedral, exclude_improper=exclude_improper)
        rdf = {}
        for pair in pairs:
            rdf[(pair[0], pair[1])] = np.zeros((pair_nbins_dict[(pair[0], pair[1])],2))
        if len(bonds) != 0:
            bonddist = {}
            for bondtype in bondtypes:
                bonddist[bondtype] = np.zeros((bond_nbins_dict[bondtype],2))
        if len(angles) != 0:
            angletypes = np.unique(angles[:,1])
            angledist = {}
            for angletype in angletypes:
                angledist[angletype] = np.zeros((angle_nbins,2))
        if len(dihedrals) != 0:
            dihedraltypes = np.unique(dihedrals[:,1])
            dihedraldist = {}
            for dihedraltype in dihedraltypes:
                dihedraldist[dihedraltype] = np.zeros((dihedral_nbins,2))
        if len(impropers) != 0:
            impropertypes = np.unique(impropers[:,1])
            improperdist = {}
            for impropertype in impropertypes:
                improperdist[impropertype] = np.zeros((improper_nbins,2))

        ### calculate target functions: partial rdf, ditribution of bond, angle, dihedral and improper
        for i in range(len(alltime)):
            d.load_traj(i)
            d.unwrap()
            TYPE, X, Y, Z = d.map("type", "xu", "yu", "zu")
            idmapping_atom = d.idtoindex_atom()
            _, _, _, _, atoms = d.snapshot()
            ### calculate partial rdf
            d.wrap()
            ne.set_snapshot(d)
            set_cutoff = set(pair_cutoff_dict.values())
            set_nbins = set(pair_nbins_dict.values())
            if len(set_cutoff) == 1 and len(set_nbins) == 1:
                cutoff = set_cutoff.pop()
                nbins = set_nbins.pop()
                tmprdf = ne.radial_distribution_function_by_type(cutoff, nbins)
                for pair in pairs:
                    rdf[(pair[0], pair[1])] += tmprdf[(pair[0], pair[1])]
            else:
                for pair in pairs:
                    cutoff = pair_cutoff_dict[(pair[0], pair[1])]
                    nbins = pair_nbins_dict[(pair[0], pair[1])]
                    center_mask = atoms[:,TYPE] == pair[0]
                    neighbor_mask = atoms[:,TYPE] == pair[1]
                    tmprdf = ne.radial_distribution_function(cutoff, nbins, center_mask, neighbor_mask)
                    tmprdf[:,0] += cutoff - tmprdf[-1,0]
                    rdf[(pair[0], pair[1])] += tmprdf
            ### calculate bond distribution
            if len(bonds) != 0:
                for bondtype in bondtypes:
                    cutofflo = bond_cutofflo_dict[bondtype]
                    cutoffhi = bond_cutoffhi_dict[bondtype]
                    nbins = bond_nbins_dict[bondtype]
                    bonddist[bondtype] += self._bond_length_distribution(cutofflo, cutoffhi, nbins, bonds[bonds[:,1]==bondtype], atoms[:,[X,Y,Z]], idmapping_atom)
            ### calculate angle distribution
            if len(angles) != 0:
                for angletype in angletypes:
                    angledist[angletype] += self._angle_distribution(angle_nbins, angles[angles[:,1]==angletype], atoms[:,[X,Y,Z]], idmapping_atom)
            ### calculate dihedral distribution
            if len(dihedrals) != 0:
                for dihedraltype in dihedraltypes:
                    dihedraldist[dihedraltype] += self._dihedral_distribution(dihedral_nbins, dihedrals[dihedrals[:,1]==dihedraltype], atoms[:,[X,Y,Z]], idmapping_atom)
            ### calculate improper distribution
            if len(impropers) != 0:
                for impropertype in impropertypes:
                    improperdist[impropertype] += self._improper_distribution(improper_nbins, impropers[impropers[:,1]==impropertype], atoms[:,[X,Y,Z]], idmapping_atom)
 
        ### prepare return
        ### rdf
        all_distributions = []
        for pair in pairs:
            rdf[(pair[0], pair[1])] /= len(alltime)
        all_distributions.append(rdf)
        ### bonds
        if len(bonds) != 0:
            for bondtype in bondtypes:
                bonddist[bondtype] /= len(alltime)
            all_distributions.append(bonddist)
        else:
            all_distributions.append({})
        ### angles
        if len(angles) != 0:
            for angletype in angletypes:
                angledist[angletype] /= len(alltime)
            all_distributions.append(angledist)
        else:
            all_distributions.append({})
        ### dihedrals
        if len(dihedrals) != 0:
            for dihedraltype in dihedraltypes:
                dihedraldist[dihedraltype] /= len(alltime)
            all_distributions.append(dihedraldist)
        else:
            all_distributions.append({})
        ### impropers
        if len(impropers) != 0:
            for impropertype in impropertypes:
                improperdist[impropertype] /= len(alltime)
            all_distributions.append(improperdist)
        else:
            all_distributions.append({})

        return all_distributions
