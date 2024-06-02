from copy import deepcopy
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

from ..force_fields.lammps_potential import lammps_potential

class potential:
    def __init__(self, Vtype, Vstyle, kBT=1.0, ibi_damping=0.2):
        '''
        Vtype:
            "pair", "bond", "angle", "dihedral", or "imporper"
        '''
        self.lammps_pot = lammps_potential()
        self.allVstyle = { \
            "pair"    : ["none"] + list(self.lammps_pot.potential['pair'].keys()), \
            "bond"    : ["none"] + list(self.lammps_pot.potential['bond'].keys()), \
            "angle"   : ["none"] + list(self.lammps_pot.potential['angle'].keys()), \
            "dihedral": ["none"] + list(self.lammps_pot.potential['dihedral'].keys()), \
            "improper": ["none"] + list(self.lammps_pot.potential['improper'].keys()), \
            }
        self.Vtype = None
        self.Vstyle = None
        self.kBT = kBT
        self.ibi_damping = ibi_damping
        ### weight for potential fitting
        self.weight = None
        ### initial guess of the potential
        self.guess = None
        ### target pressure
        self.target_pressure = None
        ### keep a record of input distribution and fitting results of the last fit (V or popt)
        self.x = None
        self.P = None
        self.V = None
        self.para_dict = {}
        self.fitmask = None
        ### boltzmann inversion tool
        self.bi_tool = potential_boltzmann_inversion()

        if Vtype in self.allVstyle.keys():
            self.Vtype = Vtype
        else:
            raise ValueError

        if Vstyle in self.allVstyle[self.Vtype]:
            self.Vstyle = Vstyle
        else:
            raise ValueError

    def initialize(self, distribution, weight=None, guess=None, minPratio=1e-3):
        """
        Input 'weight' sets the weight for potential fitting using the member function 'fit_potential',
        if the potential style is not "table".
        This fitting weight will also be used for potential update.

        For "table" style, the potential will be smoothed
        """
        self.x = deepcopy(distribution[:,0])
        self.P = deepcopy(distribution[:,1])
        ### deal with input fitting weight
        if weight is None:
            self.weight = None
        elif isinstance(weight, np.ndarray) and weight.shape == (len(self.x), 2):
            try:
                np.testing.assert_array_almost_equal(weight[:,0], self.x, decimal=5)
            except:
                raise RuntimeError("The first dimension of fitting weight should be the same as the the first dimension of distribution.")
            self.weight = deepcopy(weight[:,1])
        else:
            raise RuntimeError("Invalid fitting weight. Should be 'None' or a 2d numpy array.")
        ### deal with input guess potential
        if guess is None:
            self.guess = None
        elif isinstance(guess, np.ndarray) and guess.shape == (len(self.x), 2):
            try:
                np.testing.assert_array_almost_equal(guess[:,0], self.x, decimal=5)
            except:
                raise RuntimeError("The first dimension of guess potential should be the same as the the first dimension of distribution.")
            self.guess = deepcopy(guess[:,1])
        else:
            raise RuntimeError("Invalid fitting guess. Should be 'None' or a 2d numpy array.")

        if self.Vstyle == "table":
            if self.guess is None:
                x, V = self.bi_tool.boltzmann_inversion(distribution, kBT=self.kBT, minPratio=minPratio, style=self.Vtype)
                if self.Vtype == "pair":
                    V -= V[-1]        ### for pair potential, V[x->inf] = 0
                else:
                    V -= np.min(V)    ### for other potential (bond, angle, dihedral, improper), min(V(x)) = 0
                self.V = deepcopy(V)
            else:
                self.V = deepcopy(self.guess)
        elif self.Vstyle == "none":
            self.V = np.zeros(len(self.x))
        else:
            x = deepcopy(self.x)
            P = deepcopy(self.P)
            minP = np.max(P) * minPratio
            P[P < minP] = minP
            if self.guess is None:
                V = - self.kBT * np.log(P)
                if self.Vtype == "pair":
                    V -= V[-1]        ### for pair potential, V[x->inf] = 0
                else:
                    V -= np.min(V)    ### for other potential (bond, angle, dihedral, improper), min(V(x)) = 0
            else:
                V = deepcopy(self.guess)
            ### fit within reasonable range
            if self.Vtype == "dihedral":
                fitmask = np.ones(len(P), dtype=bool)
            else:
                fitmask = P > minP
            x_fit = x[fitmask]
            V_fit = V[fitmask]
            if self.weight is None:
                weight_fit = None
                if self.Vtype == "dihedral":
                    weight_fit = np.ones(len(P))
                    weight_fit[P <= minP] = 0.01
            else:
                weight_fit = self.weight[fitmask]
            self.fit_potential(x_fit, V_fit, weight_fit)
            self.fitmask = deepcopy(fitmask)

    def update_potential(self, current_distribution, target_distribution, minPratio=1e-3, max_dV=0.01, smooth_deltaV=True, w=5, p=3, current_pressure=None, pressure_correction_scale=0.1, debug=None):
        """
        """
        self.P = deepcopy(current_distribution[:,1])

        if self.Vstyle != "none":
            if self.Vtype == "dihedral":
                periodic = True
            else:
                periodic = False

            x, V = self.bi_tool.update_potential(self.x, self.V, current_distribution, target_distribution, \
                kBT=self.kBT, damping=self.ibi_damping, minPratio=minPratio, max_dV=max_dV, smooth_deltaV=smooth_deltaV, w=w, p=p, periodic=periodic, debug=debug)

            if self.Vstyle == "table": 
                if self.Vtype == "pair":
                    V -= V[-1]                ### for pair potential, V[x->inf] = 0
                    ### for pressure correction
                    if self.target_pressure is not None:
                        dp = current_pressure - self.target_pressure
                        A = -np.sign(dp) * pressure_correction_scale * self.kBT * min(1.0, 0.0003*np.abs(dp))
                        dV = A * (1 - x/max(x))
                        V += dV
                        if debug is not None:
                            np.savetxt("%s/pressure_correct_%s" % (debug.split('/')[0], debug.split('/')[1]), np.column_stack((x, dV, V)), header="dp = %f, A = %f\nx\tdV\tV" % (dp, A))
                else:
                    V -= np.min(V)            ### for other potential (bond, angle, dihedral, improper), min(V(x)) = 0
                self.V = deepcopy(V)
            else:
                x_fit = x[self.fitmask]
                V_fit = V[self.fitmask]
                if self.Vtype == "pair":
                    V_fit -= V_fit[-1]        ### for pair potential, V[x->inf] = 0
                    ### for pressure correction
                    if self.target_pressure is not None:
                        dp = current_pressure - self.target_pressure
                        A = -np.sign(dp) * pressure_correction_scale * self.kBT * min(1.0, 0.0003*np.abs(dp))
                        dV = A * (1 - x_fit/max(x_fit))
                        V_fit += dV
                        if debug is not None:
                            np.savetxt("%s/pressure_correct_%s" % (debug.split('/')[0], debug.split('/')[1]), np.column_stack((x_fit, dV, V_fit)), header="dp = %f, A = %f\nx\tdV\tV" % (dp, A))
                else:
                    V_fit -= np.min(V_fit)    ### for other potential (bond, angle, dihedral, improper), min(V(x)) = 0
                if self.weight is None:
                    weight_fit = None
                else:
                    weight_fit = self.weight[self.fitmask]
                self.fit_potential(x_fit, V_fit, weight_fit)

    def smooth_potential(self, s=0.5, w=5, p=3, mode="spline", periodic=False):
        """
        "mode" should be "spline" or "savgol".
        When mode="spline", s (smoothing condition) is used by scipy.interpolate.splrep, w is ignored.
        When mode="savgol", w (the ratio of window length to data range) is used to calculate window length for scipy.signal.savgol_filter
        Input "p" is the order of the polynomial used to fit the potential.
        Input "periodic" indicate whether the potential is periodic (like dihedral)
        """
        if mode == "none":
            return None
        elif mode == "spline":
            weight = np.ones(len(self.x))
            mask = self.V > 1.0
            weight[mask] = 1.0 / self.V[mask]
            tck = interpolate.splrep(self.x, self.V, w=weight, s=s, k=p)
            V = interpolate.splev(self.x, tck, der=0)
        elif mode == "savgol":
            if periodic:
                V = savgol_filter(self.V, window_length=w, polyorder=p, mode="wrap")
            else:
                V = savgol_filter(self.V, window_length=w, polyorder=p, mode="interp")
        else:
            raise ValueError("Invalid smooth mode")
        
        if self.Vtype == "pair":
            V -= V[-1]                ### for pair potential, V[x->inf] = 0
        else:
            V -= np.min(V)            ### for other potential (bond, angle, dihedral, improper), min(V(x)) = 0
        self.V = deepcopy(V)

    def write_tabulated_potential(self, filename):
        """
        Write tabulated potential obtained by iterative boltzmann inversion.
        It can also generate a tabulated potential according to the analytical form.
        """
        self.bi_tool.write_tabulated_potential(self.x, self.V, filename, self.Vtype)

    def read_tabulated_potential(self, filename):
        """
        Read tabulated potential
        """
        x, V = self.bi_tool.read_tabulated_potential(filename)
        self.x = deepcopy(x)
        self.V = deepcopy(V)

    def fit_potential(self, x_fit, V_fit, weight=None):
        """
        Fit potential curve
        """
        para_dict = {}
        V_evaluate = None
        if weight is None:
            weight = np.ones(len(x_fit))
        else:
            ### weight should no be 0. modify the weight is it is 0
            zeromask = weight == 0
            weight[zeromask] = np.min(weight[~zeromask]) * 1e-3
        para_dict, V_evaluate = self.lammps_pot.fit_and_evaluate(self.Vtype, self.Vstyle, self.x, x_fit, V_fit, weight)
        self.para_dict = deepcopy(para_dict)
        self.V = deepcopy(V_evaluate)


class potential_boltzmann_inversion:
    def __init__(self):
        pass

    def boltzmann_inversion(self, distribution, kBT=1.0, minPratio=1e-3, style="pair", maxVratio=1000, smooth=True):
        """
        Perform Boltzmann inversion on some distribution. Output potential V(x) follows:
            V(X) = -ln(P(x))
        where x is the coordinate, P(x) is probability.
        In case of pairwise interaction, P(x) is radial distribution function, x is distance.
        In case of bond interaction, P(x) is the distribution of bond length, x is the bond length.
        In case of angle interaction, P(x) should be P(x)/sin(x), x is the angle.
        In case of dihedral and improper interaction, P(x) is the distribution of dihedral/improper, x is the dihedral/improper.

        Parameters:
            distribution: 2d numpy array (double) with shape = (N,2)
                Distribution of something (pairwise distance, bond length, ...).
                1st column is the coordinate, which is also the coordinate of the corresponding potential.
                2nd column is the probability.

            kBT: float
                Boltzmann constant * temperature

            minPratio: float
                set the minimum P value by (minPratio * maximum P) to prevent log(0) when calculating potential

            style: string
                Choose from "pair", "bond", "angle", "dihedral" and "imporper"
                For "pair", the potential will diverge on the left.
                For "bond", the potential will diverge on the left and right.
                For "angle" and "improper" choices, the potential will diverge when necessary.
                For "dihedral", the potential will not diverge.
                Diverge on the left means: V(x->xmin) -> inf when P(x->xmin) = 0. V = a * (x-xmin)^(-12) is used to calculate the diverging potential.
                Diverge on the right means: V(x->xmax) -> inf when P(x->xmax) = 0. V = a / (xmax-x) is used to calculate the diverging potential.

            maxVratio: float

            smoooth: boolean

        Return:

        """
        if style == "pair":
            diverge_left = True
            diverge_right = False
        elif style == "bond":
            diverge_left = True
            diverge_right = True
        else:
            diverge_left = False
            diverge_right = False

        x = distribution[:,0]
        P = distribution[:,1]
        minP = np.max(P) * minPratio
        maxV = -np.log(minP) * maxVratio
        P[P < minP] = minP      ### avoid P = 0 when calculating ln(P)
        V = - kBT * np.log(P)

        if style == "angle" or style == "improper":
            if P[0] == minP:
                diverge_left = True
            if P[-1] == minP:
                diverge_right = True

        if diverge_left:
            maskid = np.where(P > minP)
            fromid = np.min(maskid) - 1
            if fromid > 0:
                a = - kBT * np.log(minP) * (x[fromid] - np.min(x))**12
                V[1:fromid + 1] = a / (x[1:fromid + 1] - np.min(x))**12
                minx = (a / maxV)**(1.0 / 12) + np.min(x)
                V[x < minx] = maxV

        if diverge_right:
            maskid = np.where(P > minP)
            fromid = np.max(maskid) + 1
            if fromid < len(x):
                a = - kBT * np.log(minP) * (np.max(x) - x[fromid - 1])**12
                V[fromid:-1] = a / (np.max(x) - x[fromid:-1])**12
                maxx = np.max(x) - (a / maxV)**(1.0 / 12)
                V[x > maxx] = maxV

        if smooth:
        ### smooth the potential by spline
            if style == "dihedral":
                tck = interpolate.splrep(x, V, s=1, k=3, per=1)
            else:
                tck = interpolate.splrep(x, V, s=1, k=3)
            V = interpolate.splev(x, tck, der=0)

        return x, V

    def update_potential(self, x, V, current_distribution, target_distribution, kBT=1.0, damping=1.0, minPratio=1e-3, max_dV=0.01, smooth_deltaV=True, w=5, p=3, periodic=False, debug=None):
        """
        Update the potential V(x) based on current distribution and target distribution.
            V(x) += deltaV(x)
            deltaV = kBT * damping * ln(current_P(x)/target_P(x))
        where x is the coordinate, current_P(x) and target_P(X) are probability of current and target distribution, respectively.
        For P(x) of different interaction, see docstring of function boltzmann_inversion.

        Parameters:
            current_distribution: 2d numpy array (double) with shape = (N,2)
                Distribution of something (pairwise distance, bond length, ...), obtained from current simulation snapshots.
                1st column is the coordinate, x, which is also the coordinate of the corresponding potential.
                2nd column is the probability, current_P(x).

            target_distribution: 2d numpy array (double) with shape = (N,2)
                Distribution of something (pairwise distance, bond length, ...), obtained from target simulation snapshots.
                1st column is the coordinate, x, which is also the coordinate of the corresponding potential.
                2nd column is the probability, target_P(x).

        """
        np.testing.assert_array_almost_equal(x, current_distribution[:,0], decimal=5)
        np.testing.assert_array_almost_equal(x, target_distribution[:,0], decimal=5)
        #if not np.array_equal(x, current_distribution[:,0]):
        #    print("Current distribution and current potential don't share the same coordinate.")
        #    raise RuntimeError
        #if not np.array_equal(x, target_distribution[:,0]):
        #    print("Target distribution and current potential don't share the same coordinate.")
        #    raise RuntimeError
        current_P = deepcopy(current_distribution[:,1])
        target_P = deepcopy(target_distribution[:,1])

        ### reset zero values (and values that are too small) in current and target distribution
        minP = np.min([np.max(current_P), np.max(target_P)]) * minPratio
        current_P[current_P < minP] = minP
        target_P[target_P < minP] = minP

        ### calculate deltaV. Cap the deltaV with max_dV.
        deltaV = kBT * damping * np.log(current_P / target_P)
        deltaV[deltaV > max_dV] = max_dV
        deltaV[deltaV < -max_dV] = -max_dV

        if debug is not None:
            np.savetxt("%s/nosmooth_%s" % (debug.split('/')[0], debug.split('/')[1]), np.column_stack((x, deltaV, V)), header="x\tdV\tV")

        ### smooth deltaV (according to mannual of VOTCA (http://doc.votca.org/manual.pdf), it is better to smooth
        ### the change of potential deltaV than the updated potential)
        if smooth_deltaV:
            if periodic:
                deltaV = savgol_filter(deltaV, window_length=w, polyorder=p, mode="wrap")
            else:
                deltaV = savgol_filter(deltaV, window_length=w, polyorder=p, mode="interp")
        
            if debug is not None:
                np.savetxt("%s/smooth_%s" % (debug.split('/')[0], debug.split('/')[1]), np.column_stack((x, deltaV, V)), header="x\tdV\tV")

        return x, V + deltaV

    def write_tabulated_potential(self, x, V, filename, Vtype):
        """
        Write tabulated potential
        """
        if Vtype == "pair":
            f = open(filename, "w")
            f.write("#IBI pair potential file\n\n")
            f.write("IBI\n")
            f.write("N %d R %g %g\n\n" % (len(V), np.min(x), np.max(x)))
            f.close()
        elif Vtype == "bond":
            f = open(filename, "w")
            f.write("#IBI bond potential file\n\n")
            f.write("IBI\n")
            f.write("N %d\n\n" % len(V))
            f.close()
        elif Vtype == "angle":
            f = open(filename, "w")
            f.write("#IBI angle potential file\n\n")
            f.write("IBI\n")
            f.write("N %d\n\n" % len(V))
            f.close()
        elif Vtype == "dihedral":
            f = open(filename, "w")
            f.write("#IBI dihedral potential file\n\n")
            f.write("IBI\n")
            f.write("N %d\n\n" % len(V))
            f.close()
        else:
            print("No support for pontential type %s." % Vtype)
            raise ValueError

        index = np.arange(1, len(V) + 1)
        ### calculate force
        force_left = - (V[1:] - V[:-1]) / (x[1:] - x[:-1])
        force_left = force_left[:-1]
        force_right = - (V[2:] - V[1:-1]) / (x[2:] - x[1:-1])
        force = 0.5 * (force_left + force_right)
        force = np.insert(force, 0, force[0])
        force = np.append(force, force[-1])

        f = open(filename, "ab")
        np.savetxt(f, np.column_stack((index, x, V, force)), fmt="%d %.4f %.4f %.4f")
        f.close()

    def read_tabulated_potential(self, filename):
        """
        Read tabulated potential
        """
        data = np.loadtxt(filename, usecols=(1,2), skiprows=4)
        x = data[:,0]
        V = data[:,1]
        return x, V
