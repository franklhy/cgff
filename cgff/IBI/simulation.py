from copy import deepcopy
import numpy as np

class simulation:
    def __init__(self, md_setup_file):
        self.lammpsname = None
        self.scriptname = "in.IBI"
        self.ngpu = 0
        self.units = "real"
        self.md_style = "nvt"
        self.coulumbic = None
        self.debye_length = None
        self.dielectric = None
        self.temp = None
        self.tdamp = None
        self.pres = None
        self.pdamp = None
        self.special_bonds = [0.0, 0.0, 0.0]
        self.timestep = 1.0
        self.dataname = None
        self.replicate = 1
        self.ndump = None
        self.dumpname = None
        self.relax_nvelimit = 0.05
        self.prod_nvelimit = 0.5
        self.relaxrun = None
        self.prodrun = None
        with open(md_setup_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("#")[0]
                if line:
                    first, second = line.split()
                    if first == "lammpsname":
                        self.lammpsname = second
                    elif first == "scriptname":
                        self.scriptname = second
                    elif first == "ngpu":
                        self.ngpu = int(second)
                    elif first == "units":
                        self.units = second
                    elif first == "md_style":
                        self.md_style = second
                    elif first == "coulumbic":
                        self.coulumbic = second
                    elif first == "debye_length":
                        self.debye_length = float(second)
                    elif first == "dielectric":
                        self.dielectric = float(second)
                    elif first == "temp":
                        self.temp = float(second)
                    elif first == "tdamp":
                        self.tdamp = float(second)
                    elif first == "pres":
                        self.pres = float(second)
                    elif first == "pdamp":
                        self.pdamp = float(second)
                    elif first == "special_bonds":
                        self.special_bonds = [float(s) for s in second.split(',')]
                    elif first == "timestep":
                        self.timestep = float(second)
                    elif first == "dataname":
                        self.dataname = second
                    elif first == "replicate":
                        self.replicate = int(second)
                    elif first == "ndump":
                        self.ndump = int(second)
                    elif first == "dumpname":
                        self.dumpname = second
                    elif first == "relax_nvelimit":
                        self.relax_nvelimit = float(second)
                    elif first == "prod_nvelimit":
                        self.prod_nvelimit = float(second)
                    elif first == "relaxrun":
                        self.relaxrun = int(second)
                    elif first == "prodrun":
                        self.prodrun = int(second)
                    else:
                        print("Wrong format in setup file for md simulation:\n")
                        print(line)
                        raise ValueError
        if self.units == "real":
            self.kBT = 0.0019872 * self.temp    ### unit: Kcal/mol, kB = 0.0019872 Kcal/mol/K
        elif self.units == "lj":
            self.kBT = 1.0 * self.temp
        else:
            print("Units %d is not supported.")
            raise ValueError
        ### the name of file recording the average pressure (used for correction of pairwise potential)
        self.pressure_file = "pressure.txt"
        self.pressure_hist_file = "pressure_hist.txt"
        self.pressure_hist_lo = -1000
        self.pressure_hist_hi = 1000
        ### dictionary of potentials, will be modified in member function prepare
        self.pair_potential = None
        self.bond_potential = None
        self.angle_potential = None
        self.dihedral_potential = None
        self.improper_potential = None
        self.pair_style = None
        self.bond_style = None
        self.angle_style = None
        self.dihedral_style = None
        self.improper_style = None
        self.pair_max_nbins = None
        self.pair_max_cutoff = None
        self.bond_max_nbins = None
        self.angle_max_nbins = None
        self.dihedral_max_nbins = None
        self.convert_dihedral_to_tablecut = None

    def prepare(self, pair_potential, bond_potential, angle_potential, dihedral_potential, improper_potential, convert_dihedral_to_tablecut=False):
        """
        *_potential: a dictionary of instances of class potential in potential.py

        """
        self.pair_potential = deepcopy(pair_potential)
        self.bond_potential = deepcopy(bond_potential)
        self.angle_potential = deepcopy(angle_potential)
        self.dihedral_potential = deepcopy(dihedral_potential)
        self.improper_potential = deepcopy(improper_potential)

        if bool(self.pair_potential.values()):
            self.pair_style = list(self.pair_potential.values())[0].Vstyle
            nbins = []
            cutoff = []
            for key in self.pair_potential.keys():
                nbins.append(len(self.pair_potential[key].x))
                cutoff.append(np.max(self.pair_potential[key].x))
            self.pair_max_nbins = np.max(nbins)
            self.pair_max_cutoff = np.max(cutoff)
        if bool(self.bond_potential.values()):
            self.bond_style = list(self.bond_potential.values())[0].Vstyle
            nbins = []
            for key in self.bond_potential.keys():
                nbins.append(len(self.bond_potential[key].x))
            self.bond_max_nbins = np.max(nbins)
        if bool(self.angle_potential.values()):
            self.angle_style = list(self.angle_potential.values())[0].Vstyle
            nbins = []
            for key in self.angle_potential.keys():
                nbins.append(len(self.angle_potential[key].x))
            self.angle_max_nbins = np.max(nbins)
        if bool(self.dihedral_potential.values()):
            self.dihedral_style = list(self.dihedral_potential.values())[0].Vstyle
            nbins = []
            for key in self.dihedral_potential.keys():
                nbins.append(len(self.dihedral_potential[key].x))
            self.dihedral_max_nbins = np.max(nbins)
        if bool(self.improper_potential.values()):
            self.improper_style = list(self.improper_potential.values())[0].Vstyle

        self.convert_dihedral_to_tablecut = convert_dihedral_to_tablecut


    def write_lammps_script(self):
        """
        """
        f = open(self.scriptname, "w")
        f.write("# input script for iterative Boltzmann inversion.\n")
        if self.ngpu > 0:
            f.write("%-16s%s\n" % ("package", "gpu %d" % self.ngpu))
        f.write("\n")

        f.write("%-16s%s\n" % ("units", self.units))
        f.write("%-16s%s\n" % ("atom_style", "full"))
        f.write("%-16s%s\n" % ("neighbor", "1.0 bin"))
        f.write("%-16s%s\n" % ("neigh_modify", "every 5 delay 0 check yes"))
        f.write("%-16s%s\n" % ("boundary", "p p p"))
        f.write("\n")

        self._write_potential_styles(f)
        f.write("%-16s%s\n" % ("read_data", "input.dat nocoeff"))
        if self.replicate > 1:
            f.write("%-16s%d %d %d\n" % ("replicate", self.replicate, self.replicate, self.replicate))
        self._write_pair_coeff(f)
        self._write_bond_coeff(f)
        self._write_angle_coeff(f)
        self._write_dihedral_coeff(f)
        self._write_improper_coeff(f)
        self._write_pair_write(f)
        self._write_bond_write(f)
        f.write("%-16s%s\n" % ("special_bonds", "lj/coul %f %f %f" % (*self.special_bonds,)))

        ### energy minimization and initial NVT relaxation to relax possible poor configurations
        f.write("\n")
        f.write("%-16s%s\n" % ("minimize", "0.0 1.0e-8 1000 100000"))
        f.write("\n")
        f.write("%-16s%s\n" % ("velocity", "all create %g 54321 dist uniform" % self.temp))
        f.write("%-16s%s\n" % ("velocity", "all zero linear"))
        f.write("%-16s%s\n" % ("velocity", "all zero angular"))
        f.write("%-16s%s\n" % ("fix", "1 all nve/limit %g" % self.relax_nvelimit))
        f.write("%-16s%s\n" % ("fix", "2 all langevin %g %g %g 12345" % (self.temp, self.temp, self.tdamp)))
        f.write("\n")
        f.write("%-16s%s\n" % ("thermo", "1000"))
        f.write("%-16s%s\n" % ("thermo_style", "custom step temp evdwl ecoul elong ebond eangle edihed ke pe etotal press density vol"))
        f.write("%-16s%s\n" % ("thermo_modify", "flush yes"))
        f.write("\n")
        f.write("%-16s%s\n" % ("timestep", "%g" % self.timestep))
        f.write("%-16s%s\n" % ("run", "%d" % self.relaxrun))
        f.write("\n")

        ### run simulation
        f.write("%-16s%s\n" % ("reset_timestep", "0"))
        f.write("%-16s%s\n" % ("unfix", "1"))
        f.write("%-16s%s\n" % ("unfix", "2"))
        if self.md_style == "nvt":
            #f.write("%-16s%s\n" % ("fix", "1 all nve"))
            f.write("%-16s%s\n" % ("fix", "1 all nve/limit %g" % self.prod_nvelimit))
            f.write("%-16s%s\n" % ("fix", "2 all langevin %g %g %g 12345" % (self.temp, self.temp, self.tdamp)))
            #f.write("%-16s%s\n" % ("unfix", "2"))
            #f.write("%-16s%s\n" % ("fix", "1 all nvt temp %g %g %g" % (self.temp, self.temp, self.tdamp)))
        elif self.md_style == "npt":
            f.write("%-16s%s\n" % ("fix", "1 all npt temp %g %g %g iso %g %g %g" % (self.temp, self.temp, self.tdamp, self.pres, self.pres, self.pdamp)))
        else:
            print("Invalid md style. You can choose: \"nvt\" or \"npt\"")
            raise ValueError
        f.write("\n")
        f.write("%-16s%s\n" % ("compute", "press all pressure thermo_temp"))
        f.write("%-16s%s\n" % ("fix", "avg all ave/time 10 %d %d c_press file %s" % (self.prodrun/10, self.prodrun, self.pressure_file)))
        f.write("%-16s%s\n" % ("fix", "hist all ave/histo 10 %d %d %f %f 20000 c_press file %s" % (self.prodrun/10, self.prodrun, self.pressure_hist_lo, self.pressure_hist_hi, self.pressure_hist_file)))
        f.write("\n")
        f.write("%-16s%s\n" % ("dump", "1 all custom %d %s id type mol x y z ix iy iz vx vy vz" % (int(self.prodrun / self.ndump), self.dumpname)))
        f.write("\n")
        f.write("%-16s%s\n" % ("run", "%d" % self.prodrun))
        f.write("\n")
        f.write("%-16s%s\n" % ("write_data", self.dataname))
        f.close()


    def _write_potential_styles(self, f):
        """
        """
        if self.ngpu >= 1:
            gpukeyword = "/gpu"
        else:
            gpukeyword = ""
        if self.coulumbic in ["long", "debye"]:
            couloverlay = "hybrid/overlay coul/%s" % self.coulumbic
            if self.coulumbic == "long":
                couloverlay += " %g " % (self.pair_max_cutoff * 2)
            elif self.coulumbic == "debye":
                if self.debye_length is None:
                    raise RuntimeError("Please set debye length")
                couloverlay += " %g %g " % (1/self.debye_length, self.debye_length * 5)
        else:
            couloverlay = ""
        if self.pair_style == "table":
            f.write("%-16s%s\n" % ("pair_style", "%stable%s linear %d" % (couloverlay, gpukeyword, self.pair_max_nbins)))
        elif self.pair_style == "lj/cut":
            f.write("%-16s%s\n" % ("pair_style", "%slj/cut%s %g" % (couloverlay, gpukeyword, self.pair_max_cutoff)))
        elif self.pair_style == "lj96/cut":
            f.write("%-16s%s\n" % ("pair_style", "%slj96/cut%s %g" % (couloverlay, gpukeyword, self.pair_max_cutoff)))
        elif self.pair_style == "lj/expand":
            f.write("%-16s%s\n" % ("pair_style", "%slj/expand%s %g" % (couloverlay, gpukeyword, self.pair_max_cutoff)))
        f.write("%-16s%s\n" % ("pair_modify", "shift yes")) 

        if self.bond_style == "none":
            f.write("%-16s%s\n" % ("bond_style", "none"))
        elif self.bond_style == "table":
            f.write("%-16s%s\n" % ("bond_style", "table linear %d" % self.bond_max_nbins))
        elif self.bond_style == "harmonic":
            f.write("%-16s%s\n" % ("bond_style", "harmonic"))
        elif self.bond_style == "class2":
            f.write("%-16s%s\n" % ("bond_style", "class2"))

        if self.angle_style == "none":
            f.write("%-16s%s\n" % ("angle_style", "none"))
        elif self.angle_style == "table":
            f.write("%-16s%s\n" % ("angle_style", "table linear %d" % self.angle_max_nbins))
        elif self.angle_style == "harmonic":
            f.write("%-16s%s\n" % ("angle_style", "harmonic"))
        elif self.angle_style == "quartic":
            f.write("%-16s%s\n" % ("angle_style", "quartic"))

        if self.dihedral_style == "none":
            f.write("%-16s%s\n" % ("dihedral_style", "none"))
        elif self.convert_dihedral_to_tablecut == True:
            f.write("%-16s%s\n" % ("dihedral_style", "table/cut linear %d" % self.dihedral_max_nbins))
        elif self.dihedral_style == "table":
            f.write("%-16s%s\n" % ("dihedral_style", "table linear %d" % self.dihedral_max_nbins))
        elif self.dihedral_style == "table/cut":
            f.write("%-16s%s\n" % ("dihedral_style", "table/cut linear %d" % self.dihedral_max_nbins))
        elif self.dihedral_style == "harmonic":
            f.write("%-16s%s\n" % ("dihedral_style", "harmonic"))
        elif self.dihedral_style == "multi/harmonic":
            f.write("%-16s%s\n" % ("dihedral_style", "multi/harmonic"))
        elif self.dihedral_style == "fourier":
            f.write("%-16s%s\n" % ("dihedral_style", "fourier"))

        if self.improper_style == "none":
            f.write("%-16s%s\n" % ("improper_style", "none"))
        elif self.improper_style == "harmonic":
            f.write("%-16s%s\n" % ("improper_style", "harmonic"))


    def _write_pair_coeff(self, f):
        """
        """
        if self.coulumbic in ["long", "debye"]:
            overlay = self.pair_style
        else:
            overlay = ""

        if self.pair_style == "table":
            for key in self.pair_potential.keys():
                atomtype1 = key[0]
                atomtype2 = key[1]
                #f.write("%-16s%s\n" % ("pair_coeff", "%d %d potential/pair_potential_%d_%d.txt IBI %g" % \
                #        (atomtype1, atomtype2, atomtype1, atomtype2, np.max(self.pair_potential[key].x))))
                f.write("%-16s%s\n" % ("pair_coeff", "%d %d %s potential/pair_potential_%d_%d.txt IBI" % \
                        (atomtype1, atomtype2, overlay, atomtype1, atomtype2)))
        elif self.pair_style == "lj/cut" or self.pair_style == "lj96/cut":
            for key in self.pair_potential.keys():
                atomtype1 = key[0]
                atomtype2 = key[1]
                para_dict = self.pair_potential[key].para_dict
                f.write("%-16s%s\n" % ("pair_coeff", "%d %d %s %g %g %g" % \
                    (atomtype1, atomtype2, overlay, para_dict['eps'], para_dict['sig'], para_dict['rcut'])))
        elif self.pair_style == "lj/expand":
            for key in self.pair_potential.keys():
                atomtype1 = key[0]
                atomtype2 = key[1]
                para_dict = self.pair_potential[key].para_dict
                f.write("%-16s%s\n" % ("pair_coeff", "%d %d %s %g %g %g %g" % \
                    (atomtype1, atomtype2, overlay, para_dict['eps'], para_dict['sig'], para_dict['delta'], para_dict['rcut'])))

        if self.coulumbic in ["long", "debye"]:
            f.write("%-16s%s\n" % ("pair_coeff", "* * coul/%s" % self.coulumbic))
            f.write("%-16s%s\n" % ("dielectric", "%g" % self.dielectric))

        if self.coulumbic == "long":
            f.write("%-16s%s\n" % ("kspace_style", "pppm 1e-4"))


    def _write_bond_coeff(self, f):
        """
        """
        if self.bond_style == "table":
            for bondtype in self.bond_potential.keys():
                f.write("%-16s%s\n" % ("bond_coeff", "%d potential/bond_potential_%d.txt IBI" % (bondtype, bondtype)))
        elif self.bond_style == "harmonic":
            for bondtype in self.bond_potential.keys():
                para_dict = self.bond_potential[bondtype].para_dict
                f.write("%-16s%s\n" % ("bond_coeff", "%d %g %g" % (bondtype, para_dict['kb'], para_dict['b0'])))
        elif self.bond_style == "class2":
            for bondtype in self.bond_potential.keys():
                para_dict = self.bond_potential[bondtype].para_dict
                f.write("%-16s%s\n" % ("bond_coeff", "%d %g %g %g %g" % (bondtype, para_dict['b0'], para_dict['k2'], para_dict['k3'], para_dict['k4'])))


    def _write_angle_coeff(self, f):
        """
        """
        if self.angle_style == "table":
            for angletype in self.angle_potential.keys():
                f.write("%-16s%s\n" % ("angle_coeff", "%d potential/angle_potential_%d.txt IBI" % (angletype, angletype)))
        elif self.angle_style == "harmonic":
            for angletype in self.angle_potential.keys():
                para_dict = self.angle_potential[angletype].para_dict
                f.write("%-16s%s\n" % ("angle_coeff", "%d %g %g" % (angletype, para_dict['ka'], para_dict['a0'])))
        elif self.angle_style == "quartic":
            for angletype in self.angle_potential.keys():
                para_dict = self.angle_potential[angletype].para_dict
                f.write("%-16s%s\n" % ("angle_coeff", "%d %g %g %g %g" % (angletype, para_dict['a0'], para_dict['k2'], para_dict['k3'], para_dict['k4'])))


    def _write_dihedral_coeff(self, f):
        """
        """
        if self.convert_dihedral_to_tablecut or self.dihedral_style == "table/cut":
            for dihedraltype in self.dihedral_potential.keys():
                f.write("%-16s%s\n" % ("dihedral_coeff", "%d aat 1.0 177 180 potential/dihedral_potential_%d.txt IBI" % (dihedraltype, dihedraltype)))
        elif self.dihedral_style == "table":
            for dihedraltype in self.dihedral_potential.keys():
                f.write("%-16s%s\n" % ("dihedral_coeff", "%d potential/dihedral_potential_%d.txt IBI" % (dihedraltype, dihedraltype)))
        elif self.dihedral_style == "harmonic":
            for dihedraltype in self.dihedral_potential.keys():
                para_dict = self.dihedral_potential[dihedraltype].para_dict
                f.write("%-16s%s\n" % ("dihedral_coeff", "%d %g %d %d" % (dihedraltype, para_dict['kd'], para_dict['d'], para_dict['n'])))
        elif self.dihedral_style == "multi/harmonic":
            for dihedraltype in self.dihedral_potential.keys():
                para_dict = self.dihedral_potential[dihedraltype].para_dict
                f.write("%-16s%s\n" % ("dihedral_coeff", "%d %g %g %g %g %g" % (dihedraltype, *para_dict['kd'])))
        elif self.dihedral_style == "fourier":
            for dihedraltype in self.dihedral_potential.keys():
                para_dict = self.dihedral_potential[dihedraltype].para_dict
                m = len(para_dict['kd'])
                f.write("%-16s%s" % ("dihedral_coeff", "%d %d" % (dihedraltype, m)))
                for i in range(m):
                    f.write(" %g %d %g" % (para_dict['kd'][i], para_dict['n'][i], para_dict['d'][i]))
                f.write("\n")


    def _write_improper_coeff(self, f):
        """
        """
        if self.improper_style == "harmonic":
            for impropertype in self.improper_potential.keys():
                para_dict = self.improper_potential[impropertype].para_dict
                f.write("%-16s%s\n" % ("improper_coeff", "%d %g %g" % (impropertype, para_dict['ki'], para_dict['x0'])))


    def _write_pair_write(self, f):
        """
        """
        if self.pair_style == "table":
            for key in self.pair_potential.keys():
                atomtype1 = key[0]
                atomtype2 = key[1]
                xmin = np.min(self.pair_potential[key].x)
                xmax = np.max(self.pair_potential[key].x)
                f.write("%-16s%s\n" % ("pair_write", "%d %d %d r %g %g potential/lmp_pair_potential_%d_%d.txt IBI" % \
                        (atomtype1, atomtype2, self.pair_max_nbins, xmin, xmax, atomtype1, atomtype2)))


    def _write_bond_write(self, f):
        """
        """
        if self.bond_style == "table":
            for bondtype in self.bond_potential.keys():
                xmin = np.min(self.bond_potential[bondtype].x)
                xmax = np.max(self.bond_potential[bondtype].x)
                f.write("%-16s%s\n" % ("bond_write", "%d %d %g %g potential/lmp_bond_potential_%d.txt IBI" % \
                    (bondtype, self.bond_max_nbins, xmin, xmax, bondtype)))
