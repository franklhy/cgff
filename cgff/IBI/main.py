import os
import time
import glob
import json
import shutil
from mpi4py import MPI
import numpy as np

from . import distributions
from . import potential
from . import simulation
from ..force_fields.ff_database import ff_database

from lammps import lammps

class main:
    def __init__(self):
        # variable set by setup()
        self.run_path = None
        self.target_data = None
        self.target_dump = None
        self.target_dist_path = None
        self.cgtypemap = None
        self.md_setup_file = None
        self.pair_setup_file = None
        self.bond_setup_file = None
        self.angle_setup_file = None
        self.dihedral_setup_file = None
        self.improper_setup_file = None
        self.logfile = None
        self.smooth_target_dist = None
        self.pair_style = None
        self.bond_style = None
        self.angle_style = None
        self.dihedral_style = None
        self.improper_style = None
        self.convert_dihedral_to_tablecut = None
        self.target_pressure = None
        self.pressure_correction_scale = None

        self.start_from_prev_step = True
        self.angle_nbins = 181
        self.dihedral_nbins = 360
        self.improper_nbins = 360
        self.inter_mol = True
        self.intra_mol = True
        self.ibi_damping_pair = 0.2
        self.ibi_damping_bond = 0.2
        self.ibi_damping_angle = 0.2
        self.ibi_damping_dihedral = 0.2
        self.ibi_damping_improper = 0.2
        self.minPratio_pair = 1e-3
        self.minPratio_bond = 1e-3
        self.minPratio_angle = 1e-3
        self.minPratio_dihedral = 1e-3
        self.minPratio_improper = 1e-3
        self.rmsd_file = "IBI.rmsd"
        ### smoothing mode： "spline", "savgol" or "none"
        ### smoothing condition s (for scipy.interpolate.splrep) for different interaction.
        ### smoothing condition w for different interaction, w * nbins = window_length (for scipy.signal.savgol_filter)
        ### Used to distribution function #(and smooth tabulated potential)
        self.smooth_mode_pair = "none"
        self.smooth_mode_bond = "none"
        self.smooth_mode_angle = "none"
        self.smooth_mode_dihedral = "none"
        self.smooth_mode_improper = "none"
        self.s_pair = 0.02
        self.s_bond = 0.02
        self.s_angle = 0.02
        self.s_dihedral = 0.02
        self.s_improper = 0.02      ### no tabulated improper potential in LAMMPS
        self.w_pair = 5
        self.w_bond = 5
        self.w_angle = 5
        self.w_dihedral = 5
        self.w_improper = 5         ### no tabulated improper potential in LAMMPS
        self.p_pair = 3
        self.p_bond = 3
        self.p_angle = 3
        self.p_dihedral = 3
        self.p_improper = 3         ### no tabulated improper potential in LAMMPS
        self.max_dV_pair = 0.1
        self.max_dV_bond = 1
        self.max_dV_angle = 0.1
        self.max_dV_dihedral = 0.1
        self.max_dV_improper = 1
        ### mpi4py
        self._comm = MPI.COMM_WORLD
        self._me = self._comm.Get_rank()
        self._nprocs = self._comm.Get_size()
        ### simulation and potential set up (will be modified by member function _read_setup)
        self.pair_cutoff_dict = {}
        self.pair_nbins_dict = {}
        self.pair_criterion_dict = {}
        self.bond_cutofflo_dict = {}
        self.bond_cutoffhi_dict = {}
        self.bond_nbins_dict = {}
        self.bond_criterion_dict = {}
        self.angle_criterion_dict = {}
        self.dihedral_criterion_dict = {}
        self.improper_criterion_dict = {}
        ### fitting weight of potentials
        self.pair_fitting_weight_dict = {}
        self.bond_fitting_weight_dict = {}
        self.angle_fitting_weight_dict = {}
        self.dihedral_fitting_weight_dict = {}
        self.improper_fitting_weight_dict = {}
        ### initial guess of potentials
        self.pair_potential_guess_dict = {}
        self.bond_potential_guess_dict = {}
        self.angle_potential_guess_dict = {}
        self.dihedral_potential_guess_dict = {}
        self.improper_potential_guess_dict = {}
        ### potentials that not to be changed during IBI (i.e. the same as the initial guess)
        self.pair_not_to_change = []
        self.bond_not_to_change = []
        self.angle_not_to_change = []
        self.dihedral_not_to_change = []
        self.improper_not_to_change = []


    def _read_setup(self, pair_setup_file=None, bond_setup_file=None, angle_setup_file=None, dihedral_setup_file=None, improper_setup_file=None):
        """
        Read setup information from setup file.

        Pair setup file should have at least 5 columns, corresponding to: atomtype1, atomtype2, cutoff, nbins, criterion.
        These specify the cutoff radius ("cutoff") and number of entries ("nbins") in the tabulated pair interaction between
        two different atom types ("atomtype1" and "atomtype2"), and criterion on RMSD of rdf for convergence ("criterion"). 

        Bond setup file should have at least 5 columns, corresponding to: bondtype, cutofflo, cutoffhi, nbins, criterion.
        These specify the range (from "cutofflo" to "cutoffhi") and number of entries ("nbins") in the tabulated bond interaction
        of different bond types ("bondtype"), and criterion on RMSD of bond length distribution for convergence ("criterion").

        Angle setup file should have at least 2 columns, corresponding to: angletype, criterion.
        These specify the convergence criterion ("criterion") on RMSD of bond angle distribution of different angle type ("angletype").
        """
        if pair_setup_file is not None:
            data = np.loadtxt(pair_setup_file, ndmin=2)
            for line in data:
                atomtype1 = int(line[0])
                atomtype2 = int(line[1])
                if atomtype1 > atomtype2:
                    tmp = atomtype1
                    atomtype1 = atomtype2
                    atomtype2 = tmp
                self.pair_cutoff_dict[(atomtype1, atomtype2)] = line[2]
                self.pair_nbins_dict[(atomtype1, atomtype2)] = int(line[3])
                self.pair_criterion_dict[(atomtype1, atomtype2)] = line[4]

        if bond_setup_file is not None:
            data = np.loadtxt(bond_setup_file, ndmin=2)
            for line in data:
                bondtype = int(line[0])
                self.bond_cutofflo_dict[bondtype] = line[1]
                self.bond_cutoffhi_dict[bondtype] = line[2]
                self.bond_nbins_dict[bondtype] = int(line[3])
                self.bond_criterion_dict[bondtype] = line[4]

        if angle_setup_file is not None:
            data = np.loadtxt(angle_setup_file, ndmin=2)
            for line in data:
                angletype = int(line[0])
                self.angle_criterion_dict[angletype] = line[1]

        if dihedral_setup_file is not None:
            data = np.loadtxt(dihedral_setup_file, ndmin=2)
            for line in data:
                dihedraltype = int(line[0])
                self.dihedral_criterion_dict[dihedraltype] = line[1]

        if improper_setup_file is not None:
            data = np.loadtxt(improper_setup_file, ndmin=2)
            for line in data:
                impropertype = int(line[0])
                self.improper_criterion_dict[impropertype] = line[1]


    def _read_typemap(self, typemapfile):
        id_to_name = ['']       ## id_to_name[0] = '' is a place holder
        if os.path.exists(typemapfile):
            with open(typemapfile, "r") as f:
                for line in f:
                    line = line.strip().split("#")[0]
                    if line:
                        words = line.split()
                        ti = int(words[0])
                        if ti == len(id_to_name):
                            if len(words[1:]) == 1:
                                name = words[1]
                            else:
                                name = tuple(words[1:])
                            id_to_name.append(name)
                        else:
                            raise RuntimeError("LAMMPS type in type mapping file should be continuous integer" + \
                                "staring from 1.")
        return id_to_name


    def _set_fitting_weight(self):
        for key in self.pair_criterion_dict.keys():
            if key not in self.pair_fitting_weight_dict.keys():
                self.pair_fitting_weight_dict[key] = None
        for key in self.bond_criterion_dict.keys():
            if key not in self.bond_fitting_weight_dict.keys():
                self.bond_fitting_weight_dict[key] = None
        for key in self.angle_criterion_dict.keys():
            if key not in self.angle_fitting_weight_dict.keys():
                self.angle_fitting_weight_dict[key] = None
        for key in self.dihedral_criterion_dict.keys():
            if key not in self.dihedral_fitting_weight_dict.keys():
                self.dihedral_fitting_weight_dict[key] = None
        for key in self.improper_criterion_dict.keys():
            if key not in self.improper_fitting_weight_dict.keys():
                self.improper_fitting_weight_dict[key] = None


    def _set_potential_guess(self):
        for key in self.pair_criterion_dict.keys():
            if key not in self.pair_potential_guess_dict.keys():
                self.pair_potential_guess_dict[key] = None
        for key in self.bond_criterion_dict.keys():
            if key not in self.bond_potential_guess_dict.keys():
                self.bond_potential_guess_dict[key] = None
        for key in self.angle_criterion_dict.keys():
            if key not in self.angle_potential_guess_dict.keys():
                self.angle_potential_guess_dict[key] = None
        for key in self.dihedral_criterion_dict.keys():
            if key not in self.dihedral_potential_guess_dict.keys():
                self.dihedral_potential_guess_dict[key] = None
        for key in self.improper_criterion_dict.keys():
            if key not in self.improper_potential_guess_dict.keys():
                self.improper_potential_guess_dict[key] = None


    def _write_simple_boltzmann_inversion(self, distribution, kBT, filename, Vtype):
        x = distribution[:,0]
        P = distribution[:,1]
        if Vtype == "pair":
            mask = P > np.max(P) * self.minPratio_pair
        elif Vtype == "bond":
            mask = P > np.max(P) * self.minPratio_bond
        elif Vtype == "angle":
            mask = P > np.max(P) * self.minPratio_angle
        elif Vtype == "dihedral":
            mask = P > np.max(P) * self.minPratio_dihedral
        elif Vtype == "improper":
            mask = P > np.max(P) * self.minPratio_improper
        x = x[mask]
        V = - kBT * np.log(P[mask])

        if Vtype == "pair":
            V -= V[-1]
        else:
            V -= np.min(V)

        bi = potential.potential_boltzmann_inversion()
        bi.write_tabulated_potential(x, V, filename, Vtype)


    def _log(self, logfile):
        '''
        save member variables to a logfile
        '''
        with open(logfile, 'w') as fp:
            vardict = {}
            for key, value in vars(self).items():
                if key[0] == "_":
                    continue
                elif isinstance(value, dict):
                    subdict = {}
                    for key_s, value_s in value.items():
                        if isinstance(value_s, np.ndarray):
                            subdict[str(key_s)] = value_s.transpose().tolist()
                        else:
                            subdict[str(key_s)] = value_s
                    vardict[key] = subdict
                else:
                    vardict[key] = value
            json.dump(vardict, fp, indent=4)


    def _addlog(self, text, logfile):
        if os.path.exists(logfile):
            f = open(logfile, "a")
        else:
            f = open(logfile, "w")
        f.write(text)
        f.close()


    def setup(self, run_path=".", target_data=None, target_dump=None, target_dist_path=None, smooth_target_dist=True, cgtypemap=None, \
        md_setup_file=None, pair_setup_file=None, bond_setup_file=None, angle_setup_file=None, dihedral_setup_file=None, improper_setup_file=None, \
        pair_style="table", bond_style="harmonic", angle_style="table", dihedral_style="table", improper_style="harmonic", \
        convert_dihedral_to_tablecut=False, \
        target_pressure=None, pressure_correction_scale=0.1,\
        logfile="log.IBI"):
        
        self.run_path = os.path.abspath(run_path)

        if target_data:
            self.target_data = os.path.abspath(target_data)
        else:
            raise RuntimeError("Parameter target_data cannot be None.")
        
        if target_dump:
            self.target_dump = os.path.abspath(target_dump)
        else:
            raise RuntimeError("Parameter target_dump cannot be None.")
        
        if target_dist_path:
            self.target_dist_path = os.path.abspath(target_dist_path)

        self.smooth_target_dist = smooth_target_dist

        if cgtypemap:
            self.cgtypemap = os.path.abspath(cgtypemap)
        else:
            raise RuntimeError("Parameter cgtypemap cannot be None.")
        
        if md_setup_file:
            self.md_setup_file = os.path.abspath(md_setup_file)
        else:
            raise RuntimeError("Parameter md_setup_file cannot be None.")
        
        if pair_setup_file:
            self.pair_setup_file = os.path.abspath(pair_setup_file)
        if bond_setup_file:
            self.bond_setup_file = os.path.abspath(bond_setup_file)
        if angle_setup_file:
            self.angle_setup_file = os.path.abspath(angle_setup_file)
        if dihedral_setup_file:
            self.dihedral_setup_file = os.path.abspath(dihedral_setup_file)
        if improper_setup_file:
            self.improper_setup_file = os.path.abspath(improper_setup_file)

        self.logfile = os.path.join(self.run_path, logfile)
        self.pair_style = pair_style
        self.bond_style = bond_style
        self.angle_style = angle_style
        self.dihedral_style = dihedral_style
        self.improper_style = improper_style
        self.convert_dihedral_to_tablecut = convert_dihedral_to_tablecut
        self.target_pressure = target_pressure
        self.pressure_correction_scale = pressure_correction_scale


    def run(self, max_iterate=30):
        """
        """
        if self._me == 0:
            dist_tool = distributions.distributions()              ### distribution tool (calculate distributions, rmsd of distributions)
            sim_tool = simulation.simulation(self.md_setup_file)        ### simulation tool (prepare LAMMPS input script)
            if sim_tool.md_style == "npt" and self.target_pressure is not None:
                print("Warning: target_pressure is set for npt ensemble, and pressure correction will not be carried out for npt ensemble.")
                self.target_pressure = None
            ff_db = ff_database()
            lammpsname = sim_tool.lammpsname
            scriptname = sim_tool.scriptname

            if os.path.exists(self.run_path):
                shutil.rmtree(self.run_path)
            os.mkdir(self.run_path)

            ### read set up files
            self._read_setup(self.pair_setup_file, self.bond_setup_file, self.angle_setup_file, self.dihedral_setup_file, self.improper_setup_file)

            ### fitting weight
            self._set_fitting_weight()

            ### initial potential guess
            self._set_potential_guess()

            ### save settings
            self._log(self.logfile)

            ### read type mapping files
            atom_typemap = self._read_typemap(self.cgtypemap + ".atom")
            bond_typemap = self._read_typemap(self.cgtypemap + ".bond")
            angle_typemap = self._read_typemap(self.cgtypemap + ".angle")
            dihedral_typemap = self._read_typemap(self.cgtypemap + ".dihedral")
            improper_typemap = self._read_typemap(self.cgtypemap + ".improper")

            ### read cg atom mass
            f = open(self.target_data, "r")
            find_mass = False
            for line in f:
                line = line.strip().split("#")[0]
                if line:
                    words = line.split()
                    if words[0] == "Masses":
                        find_mass = True
                    elif find_mass:
                        try:
                            atom_type = int(words[0])
                            atom_mass = float(words[1])
                            ff_db.atm_types[atom_typemap[atom_type]] = {'m': atom_mass}
                        except:
                            break
            if not find_mass:
                raise RuntimeError("Cannot read masses from target cg data file %s." % self.target_data)
            f.close()

            if self.target_dist_path is None:
                ### calculate target distributions (i.e. rdf, bond length distributions, ...)
                all_distributions = dist_tool.distributions(self.target_data, self.target_dump, \
                    self.pair_cutoff_dict, self.pair_nbins_dict, \
                    self.bond_cutofflo_dict, self.bond_cutoffhi_dict, self.bond_nbins_dict, \
                    self.angle_nbins, self.dihedral_nbins, self.improper_nbins, \
                    inter_mol=self.inter_mol, intra_mol=self.intra_mol)
            else:
                ### read pre-calculated target distributions
                all_distributions = dist_tool.read_distribution(self.target_dist_path)
            target_rdf = all_distributions[0]
            target_bonddist = all_distributions[1]
            target_angledist = all_distributions[2]
            target_dihedraldist = all_distributions[3]
            target_improperdist = all_distributions[4]

            os.chdir(self.run_path)
            os.mkdir("target")
            os.chdir("target")
            os.mkdir("distribution")
            os.mkdir("potential")
            ### write out the target distributions, smooth them if requried.
            for key in target_rdf.keys():
                if self.smooth_target_dist:
                    target_rdf[key] = dist_tool.smooth_distribution(target_rdf[key], s=self.s_pair, w=self.w_pair, \
                        p=self.p_pair, mode=self.smooth_mode_pair, minPratio=self.minPratio_pair)
                np.savetxt("distribution/rdf_%d_%d.txt" % (key[0], key[1]), target_rdf[key], fmt="%g", header="r\trdf(r)")
                self._write_simple_boltzmann_inversion(target_rdf[key], sim_tool.kBT, \
                    "potential/pair_potential_%d_%d.txt" % (key[0], key[1]), "pair")
            for key in target_bonddist.keys():
                if self.smooth_target_dist:
                    target_bonddist[key] = dist_tool.smooth_distribution(target_bonddist[key], s=self.s_bond, w=self.w_bond, \
                        p=self.p_bond, mode=self.smooth_mode_bond, minPratio=self.minPratio_bond)
                np.savetxt("distribution/bonddist_%d.txt" % key, target_bonddist[key], fmt="%g", header="l\tP(l)/(4*pi*l^2)")
                self._write_simple_boltzmann_inversion(target_bonddist[key], sim_tool.kBT, \
                    "potential/bond_potential_%d.txt" % key, "bond")
            for key in target_angledist.keys():
                if self.smooth_target_dist:
                    target_angledist[key] = dist_tool.smooth_distribution(target_angledist[key], s=self.s_angle, w=self.w_angle, \
                        p=self.p_angle, mode=self.smooth_mode_angle, minPratio=self.minPratio_angle)
                np.savetxt("distribution/angledist_%d.txt" % key, target_angledist[key], fmt="%g", header="theta\tP(theta)/sin(theta)")
                self._write_simple_boltzmann_inversion(target_angledist[key], sim_tool.kBT, \
                    "potential/angle_potential_%d.txt" % key, "angle")
            for key in target_dihedraldist.keys():
                if self.smooth_target_dist:
                    target_dihedraldist[key] = dist_tool.smooth_distribution(target_dihedraldist[key], s=self.s_dihedral, w=self.w_dihedral, \
                        p=self.p_dihedral, mode=self.smooth_mode_dihedral, minPratio=self.minPratio_dihedral)
                np.savetxt("distribution/dihedraldist_%d.txt" % key, target_dihedraldist[key], fmt="%g", header="theta\tP(theta)")
                self._write_simple_boltzmann_inversion(target_dihedraldist[key], sim_tool.kBT, \
                    "potential/dihedral_potential_%d.txt" % key, "dihedral")
            for key in target_improperdist.keys():
                if self.smooth_target_dist:
                    target_improperdist[key] = dist_tool.smooth_distribution(target_improperdist[key], s=self.s_improper, w=self.w_improper, \
                        p=self.p_improper, mode=self.smooth_mode_improper, minPratio=self.minPratio_improper)
                np.savetxt("distribution/improperdist_%d.txt" % key, target_improperdist[key], fmt="%g", header="theta\tP(theta)")
                self._write_simple_boltzmann_inversion(target_improperdist[key], sim_tool.kBT, \
                    "potential/improper_potential_%d.txt" % key, "improper")
            os.chdir("../")

            ### prepare potential dictionary
            pair_potential = {}
            bond_potential = {}
            angle_potential = {}
            dihedral_potential = {}
            improper_potential = {}

            os.mkdir("0")
            os.chdir("0")
            os.mkdir("distribution")
            os.mkdir("potential")
            shutil.copy2(self.target_data, "./input.dat")

            ### prepare initial potential (tabulated or specific style), and generate potential file if tabulated style is required
            ### Note that the initial tabulated potential is already smoothed when generated.
            ### pair potential
            for key in target_rdf.keys():
                if key in self.pair_not_to_change:
                    pair_potential[key] = potential.potential("pair", self.pair_style, sim_tool.kBT, ibi_damping=0)
                else:
                    pair_potential[key] = potential.potential("pair", self.pair_style, sim_tool.kBT, self.ibi_damping_pair)
                pair_potential[key].initialize(target_rdf[key], minPratio=self.minPratio_pair, \
                    weight=self.pair_fitting_weight_dict[key], guess=self.pair_potential_guess_dict[key])
                pair_potential[key].write_tabulated_potential("potential/pair_potential_%d_%d.txt" % (key[0], key[1]))
                ffdbkey = (atom_typemap[int(key[0])], atom_typemap[int(key[1])])
                if self.target_pressure is not None:
                    pair_potential[key].target_pressure = self.target_pressure
                if pair_potential[key].Vstyle == "table":
                    ff_db.par_types[ffdbkey] = {'style': 'table', 'file': "potential/pair_potential_%d_%d.txt" % (key[0], key[1]), 'keyw': 'IBI'}
                else:
                    ff_db.par_types[ffdbkey] = {'style': pair_potential[key].Vstyle}
                    ff_db.par_types[ffdbkey].update(pair_potential[key].para_dict)

            ### bond potential
            for key in target_bonddist.keys():
                if key in self.bond_not_to_change:
                    bond_potential[key] = potential.potential("bond", self.bond_style, sim_tool.kBT, ibi_damping=0)
                else:
                    bond_potential[key] = potential.potential("bond", self.bond_style, sim_tool.kBT, self.ibi_damping_bond)
                bond_potential[key].initialize(target_bonddist[key], minPratio=self.minPratio_bond, \
                    weight=self.bond_fitting_weight_dict[key], guess=self.bond_potential_guess_dict[key])
                bond_potential[key].write_tabulated_potential("potential/bond_potential_%d.txt" % key)
                ffdbkey = bond_typemap[key]
                if bond_potential[key].Vstyle == "table":
                    ff_db.bon_types[ffdbkey] = {'style': 'table', 'file': "potential/bond_potential_%d.txt" % key, 'keyw': 'IBI'}
                else:
                    ff_db.bon_types[ffdbkey] = {'style': bond_potential[key].Vstyle}
                    ff_db.bon_types[ffdbkey].update(bond_potential[key].para_dict)

            ### angle potential
            for key in target_angledist.keys():
                if key in self.angle_not_to_change:
                    angle_potential[key] = potential.potential("angle", self.angle_style, sim_tool.kBT, ibi_damping=0)
                else:
                    angle_potential[key] = potential.potential("angle", self.angle_style, sim_tool.kBT, self.ibi_damping_angle)
                angle_potential[key].initialize(target_angledist[key], minPratio=self.minPratio_angle, \
                    weight=self.angle_fitting_weight_dict[key], guess=self.angle_potential_guess_dict[key])
                angle_potential[key].write_tabulated_potential("potential/angle_potential_%d.txt" % key)
                ffdbkey = angle_typemap[key]
                if angle_potential[key].Vstyle == "table":
                    ff_db.ang_types[ffdbkey] = {'style': 'table', 'file': "potential/angle_potential_%d.txt" % key, 'keyw': 'IBI'}
                else:
                    ff_db.ang_types[ffdbkey] = {'style': angle_potential[key].Vstyle}
                    ff_db.ang_types[ffdbkey].update(angle_potential[key].para_dict)

            ### dihedral potential
            for key in target_dihedraldist.keys():
                if key in self.dihedral_not_to_change:
                    dihedral_potential[key] = potential.potential("dihedral", self.dihedral_style, sim_tool.kBT, ibi_damping=0)
                else:
                    dihedral_potential[key] = potential.potential("dihedral", self.dihedral_style, sim_tool.kBT, self.ibi_damping_dihedral)
                dihedral_potential[key].initialize(target_dihedraldist[key], minPratio=self.minPratio_dihedral, \
                    weight=self.dihedral_fitting_weight_dict[key], guess=self.dihedral_potential_guess_dict[key])
                dihedral_potential[key].write_tabulated_potential("potential/dihedral_potential_%d.txt" % key)
                ffdbkey = dihedral_typemap[key]
                if dihedral_potential[key].Vstyle == "table":
                    ff_db.dih_types[ffdbkey] = {'style': 'table', 'file': "potential/dihedral_potential_%d.txt" % key, 'keyw': 'IBI'}
                else:
                    ff_db.dih_types[ffdbkey] = {'style': dihedral_potential[key].Vstyle}
                    ff_db.dih_types[ffdbkey].update(dihedral_potential[key].para_dict)

            ### improper potential
            for key in target_improperdist.keys():
                ### table style is not available for improper in LAMMPS
                if key in self.improper_not_to_change:
                    improper_potential[key] = potential.potential("improper", self.improper_style, sim_tool.kBT, ibi_damping=0)
                else:
                    improper_potential[key] = potential.potential("improper", self.improper_style, sim_tool.kBT, self.ibi_damping_improper)
                improper_potential[key].initialize(target_improperdist[key], minPratio=self.minPratio_improper, \
                    weight=self.improper_fitting_weight_dict[key], guess=self.improper_potential_guess_dict[key])
                improper_potential[key].write_tabulated_potential("potential/improper_potential_%d.txt" % key)
                ffdbkey = improper_typemap[key]
                ff_db.imp_types[ffdbkey] = {'style': improper_potential[key].Vstyle}
                ff_db.imp_types[ffdbkey].update(improper_potential[key].para_dict)

            ### write force field database python file
            ff_db.write_forecefield_database("cg_forcefield.py")

            ### prepare lammps input script according to current force fields
            sim_tool.prepare(pair_potential, bond_potential, angle_potential, dihedral_potential, improper_potential, self.convert_dihedral_to_tablecut)
            sim_tool.write_lammps_script()
        else:
            lammpsname = None
            scriptname = None

        ### run lammps
        lammpsname = self._comm.bcast(lammpsname, root=0)
        scriptname = self._comm.bcast(scriptname, root=0)
        lmp = lammps(lammpsname)
        lmp.file(scriptname)

        loopi = 0
        finish_flag = False
        while True:
            if self._me == 0:
                ### calculate distributions (i.e. rdf, bond length distributions, ...)
                all_distributions = dist_tool.distributions(sim_tool.dataname, sim_tool.dumpname, \
                    self.pair_cutoff_dict, self.pair_nbins_dict, \
                    self.bond_cutofflo_dict, self.bond_cutoffhi_dict, self.bond_nbins_dict, \
                    self.angle_nbins, self.dihedral_nbins, self.improper_nbins, \
                    inter_mol=self.inter_mol, intra_mol=self.intra_mol)
                rdf = all_distributions[0]
                bonddist = all_distributions[1]
                angledist = all_distributions[2]
                dihedraldist = all_distributions[3]
                improperdist = all_distributions[4]

                ### write out distributions
                for key in rdf.keys():
                    #rdf[key] = dist_tool.smooth_distribution(rdf[key], w=self.w_pair, mode=self.smooth_mode_pair)
                    np.savetxt("distribution/rdf_%d_%d.txt" % (key[0], key[1]), rdf[key], fmt="%g", header="r\trdf(r)")
                for key in bonddist.keys():
                    #bonddist[key] = dist_tool.smooth_distribution(bonddist[key], w=self.w_bond, mode=self.smooth_mode_bond)
                    np.savetxt("distribution/bonddist_%d.txt" % key, bonddist[key], fmt="%g", header="l\tP(l)/(4*pi*l^2)")
                for key in angledist.keys():
                    #angledist[key] = dist_tool.smooth_distribution(angledist[key], w=self.w_angle, mode=self.smooth_mode_angle)
                    np.savetxt("distribution/angledist_%d.txt" % key, angledist[key], fmt="%g", header="theta\tP(theta)/sin(theta)")
                for key in dihedraldist.keys():
                    #dihedraldist[key] = dist_tool.smooth_distribution(dihedraldist[key], w=self.w_dihedral, mode=self.smooth_mode_dihedral)
                    np.savetxt("distribution/dihedraldist_%d.txt" % key, dihedraldist[key], fmt="%g", header="theta\tP(theta)")
                for key in improperdist.keys():
                    #improperdist[key] = dist_tool.smooth_distribution(improperdist[key], w=self.w_improper, mode=self.smooth_mode_improper)
                    np.savetxt("distribution/improperdist_%d.txt" % key, improperdist[key], fmt="%g", header="theta\tP(theta)")

                ### check convergence, and write rmsd of different distributions to a file
                rmsdf = open(self.rmsd_file, "w")
                converged = []
                for key in rdf.keys():
                    rmsd = dist_tool.rmsd(rdf[key], target_rdf[key])
                    converged.append(rmsd < self.pair_criterion_dict[key])
                    rmsdf.write("%-16s%f\n" % ("pair_%d_%d" % (key[0], key[1]), rmsd))
                for key in bonddist.keys():
                    rmsd = dist_tool.rmsd(bonddist[key], target_bonddist[key])
                    converged.append(rmsd < self.bond_criterion_dict[key])
                    rmsdf.write("%-16s%f\n" % ("bond_%d" % key, rmsd))
                for key in angledist.keys():
                    rmsd = dist_tool.rmsd(angledist[key], target_angledist[key])
                    converged.append(rmsd < self.angle_criterion_dict[key])
                    rmsdf.write("%-16s%f\n" % ("angle_%d" % key, rmsd))
                for key in dihedraldist.keys():
                    rmsd = dist_tool.rmsd(dihedraldist[key], target_dihedraldist[key])
                    converged.append(rmsd < self.dihedral_criterion_dict[key])
                    rmsdf.write("%-16s%f\n" % ("dihedral_%d" % key, rmsd))
                for key in improperdist.keys():
                    rmsd = dist_tool.rmsd(improperdist[key], target_improperdist[key])
                    converged.append(rmsd < self.improper_criterion_dict[key])
                    rmsdf.write("%-16s%f\n" % ("improper_%d" % key, rmsd))
                rmsdf.close()

                if np.all(converged):
                    print("Converged!")
                    finish_flag = True

                loopi += 1
                os.chdir("../")

                if loopi > max_iterate:
                    print("Maximum iteration %d reached. Not converged." % max_iterate)
                    finish_flag = True

            finish_flag = self._comm.bcast(finish_flag, root=0)
            if finish_flag:
                break

            if self._me == 0:
                os.mkdir("%d" % loopi)
                os.chdir("%d" % loopi)
                os.mkdir("distribution")
                os.mkdir("potential")
                os.mkdir("debug")
                if self.start_from_prev_step:
                    shutil.copy2("../%d/%s" % (loopi - 1, sim_tool.dataname), "./input.dat")
                else:
                    shutil.copy2("../0/input.dat", "./input.dat")

                ### read the average pressure
                current_pressure = None
                if self.target_pressure is not None:
                    pressure_file = open("../%d/%s" % (loopi - 1, sim_tool.pressure_file))
                    pressure_file.readline()
                    pressure_file.readline()
                    current_pressure = float(pressure_file.readline().split()[1])
                    print("current pressure of iteration no. %d is %f." % (loopi - 1, current_pressure))
                    print("target pressure is %f." % self.target_pressure)

                ### update potential (tabulated or specific style), and generate potential file if tabulated style is required
                ### pair potential
                for key in target_rdf.keys():
                    pair_potential[key].update_potential(rdf[key], target_rdf[key], \
                        minPratio=self.minPratio_pair, max_dV=self.max_dV_pair, smooth_deltaV=True, w=self.w_pair, p=self.p_pair, \
                        current_pressure=current_pressure, pressure_correction_scale=self.pressure_correction_scale, \
                        debug="debug/pair_dV_%d_%d.txt" % (key[0], key[1]))
                    #pair_potential[key].write_tabulated_potential("potential/pair_potential_nosmooth_%d_%d.txt" % (key[0], key[1]))
                    #pair_potential[key].smooth_potential(s=self.s_pair, w=self.w_pair, p=self.p_pair, mode=self.smooth_mode_pair)
                    pair_potential[key].write_tabulated_potential("potential/pair_potential_%d_%d.txt" % (key[0], key[1]))
                    if pair_potential[key].Vstyle != "table":
                        ffdbkey = (atom_typemap[int(key[0])], atom_typemap[int(key[1])])
                        ff_db.par_types[ffdbkey].update(pair_potential[key].para_dict)

                ### bond potential
                for key in target_bonddist.keys():
                    bond_potential[key].update_potential(bonddist[key], target_bonddist[key], \
                        minPratio=self.minPratio_bond, max_dV=self.max_dV_bond, smooth_deltaV=True, w=self.w_bond, p=self.p_bond, \
                        debug="debug/bond_dV_%d.txt" % key)
                    #bond_potential[key].write_tabulated_potential("potential/bond_potential_nosmooth_%d.txt" % key)
                    #bond_potential[key].smooth_potential(s=self.s_bond, w=self.w_bond, p=self.p_bond, mode=self.smooth_mode_bond)
                    bond_potential[key].write_tabulated_potential("potential/bond_potential_%d.txt" % key)
                    if bond_potential[key].Vstyle != "table":
                        ffdbkey = bond_typemap[key]
                        ff_db.bon_types[ffdbkey].update(bond_potential[key].para_dict)

                ### angle potential
                for key in target_angledist.keys():
                    angle_potential[key].update_potential(angledist[key], target_angledist[key], \
                        minPratio=self.minPratio_angle, max_dV=self.max_dV_angle, smooth_deltaV=True, w=self.w_angle, p=self.p_angle, \
                        debug="debug/angle_dV_%d.txt" % key)
                    #angle_potential[key].write_tabulated_potential("potential/angle_potential_nosmooth_%d.txt" % key)
                    #angle_potential[key].smooth_potential(s=self.s_angle, w=self.w_angle, p=self.p_angle, mode=self.smooth_mode_angle)
                    angle_potential[key].write_tabulated_potential("potential/angle_potential_%d.txt" % key)
                    if angle_potential[key].Vstyle != "table":
                        ffdbkey = angle_typemap[key]
                        ff_db.ang_types[ffdbkey].update(angle_potential[key].para_dict)

                ### dihedral potential
                for key in target_dihedraldist.keys():
                    dihedral_potential[key].update_potential(dihedraldist[key], target_dihedraldist[key], \
                        minPratio=self.minPratio_dihedral, max_dV=self.max_dV_dihedral, smooth_deltaV=True, w=self.w_dihedral, p=self.p_dihedral, \
                        debug="debug/dihedral_dV_%d.txt" % key)
                    #dihedral_potential[key].write_tabulated_potential("potential/dihedral_potential_nosmooth_%d.txt" % key)
                    #dihedral_potential[key].smooth_potential(s=self.s_dihedral, w=self.w_dihedral, p=self.p_dihedral, mode=self.smooth_mode_dihedral)
                    dihedral_potential[key].write_tabulated_potential("potential/dihedral_potential_%d.txt" % key)
                    if dihedral_potential[key].Vstyle[:5] != "table":
                        ffdbkey = dihedral_typemap[key]
                        ff_db.dih_types[ffdbkey].update(dihedral_potential[key].para_dict)

                ### improper potential
                for key in target_improperdist.keys():
                    ### table style is not available for improper in LAMMPS
                    improper_potential[key].update_potential(improperdist[key], target_improperdist[key], \
                        minPratio=self.minPratio_improper, max_dV=self.max_dV_improper, smooth_deltaV=True, w=self.w_improper, p=self.p_improper, \
                        debug="debug/improper_dV_%d.txt" % key)
                    improper_potential[key].write_tabulated_potential("potential/improper_potential_%d.txt" % key)
                    ffdbkey = improper_typemap[key]
                    ff_db.imp_types[ffdbkey].update(improper_potential[key].para_dict)

                ### write force field database python file
                ff_db.write_forecefield_database("cg_forcefield.py")

                ### prepare lammps input script according to current force fields
                sim_tool.prepare(pair_potential, bond_potential, angle_potential, dihedral_potential, improper_potential)
                sim_tool.write_lammps_script()

            loopi = self._comm.bcast(loopi, root=0)
            ### run lammps
            lmp = lammps(lammpsname)
            lmp.file(scriptname)

        lmp.close()
