import numpy as np
from scipy.optimize import curve_fit

class lammps_potential:
    def __init__(self):
        self.potential = {'pair': {}, 'bond': {}, 'angle': {}, 'dihedral': {}, 'improper': {}}

        self.potential['pair']['table'] = {'file': '', 'keyw': ''}
        self.potential['pair']['lj/cut'] = {'eps': 0, 'sig': 0, 'rcut': 0}
        self.potential['pair']['lj96/cut'] = {'eps': 0, 'sig': 0, 'rcut': 0}
        self.potential['pair']['lj/expand'] = {'eps': 0, 'sig': 0, 'delta': 0, 'rcut': 0}

        self.potential['bond']['table'] = {'file': '', 'keyw': ''}
        self.potential['bond']['harmonic'] = {'kb': 0, 'b0': 0}
        self.potential['bond']['class2'] = {'b0': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'C': 0}  ### C is not included in LAMMMPS

        self.potential['angle']['table'] = {'file': '', 'keyw': ''}
        self.potential['angle']['harmonic'] = {'ka': 0, 'a0': 0}
        self.potential['angle']['quartic'] = {'a0': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'C': 0}  ### C is not included in LAMMMPS

        self.potential['dihedral']['table'] = {'file': '', 'keyw': ''}
        self.potential['dihedral']['harmonic'] = {'kd': 0, 'd': 0, 'n': 0}
        self.potential['dihedral']['multi/harmonic'] = {'kd': [0, 0, 0, 0, 0]}
        self.potential['dihedral']['fourier'] = {'kd': [0, 0, 0, 0], 'n': [0, 0, 0, 0], 'd': [0, 0, 0, 0]}

        self.potential['improper']['harmonic'] = {'ki': 0, 'x0': 0}

    def fit_and_evaluate(self, Vtype, Vstyle, x, x_fit, V_fit, weight):
        if Vtype == 'pair':
            if Vstyle == 'lj/cut':
                V = self._fit_pair_ljcut(x, x_fit, V_fit, weight)
            elif Vstyle == 'lj96/cut':
                V = self._fit_pair_lj96cut(x, x_fit, V_fit, weight)
            elif Vstyle == 'lj/expand':
                V = self._fit_pair_ljexpand(x, x_fit, V_fit, weight)
        elif Vtype == 'bond':
            if Vstyle == 'harmonic':
                V = self._fit_bond_harmonic(x, x_fit, V_fit, weight)
            elif Vstyle == 'class2':
                V = self._fit_bond_class2(x, x_fit, V_fit, weight)
        elif Vtype == 'angle':
            if Vstyle == 'harmonic':
                V = self._fit_angle_harmonic(x, x_fit, V_fit, weight)
            elif Vstyle == 'quartic':
                V = self._fit_angle_quartic(x, x_fit, V_fit, weight)
        elif Vtype == 'dihedral':
            if Vstyle == 'harmonic':
                V = self._fit_dihedral_harmonic(x, x_fit, V_fit, weight)
            elif Vstyle == 'multi/harmonic':
                V = self._fit_dihedral_multiharmonic(x, x_fit, V_fit, weight)
            elif Vstyle == 'fourier':
                V = self._fit_dihedral_fourier(x, x_fit, V_fit, weight)
        elif Vtype == 'improper':
            if Vstyle == 'harmonic':
                V = self._fit_improper_harmonic(x, x_fit, V_fit, weight)

        return self.potential[Vtype][Vstyle], V

    def evaluate(self, Vtype, Vstyle, x, para_dict):
        '''
        x: 1d numpy array
        '''
        if Vtype == 'pair':
            if Vstyle == 'lj/cut':
                V = self._pair_ljcut(x, para_dict['eps'], para_dict['sig'], para_dict['rcut'])
            elif Vstyle == 'lj96/cut':
                V = self._pair_lj96cut(x, para_dict['eps'], para_dict['sig'], para_dict['rcut'])
            elif Vstyle == 'lj/expand':
                V = self._pair_ljexpand(x, para_dict['eps'], para_dict['sig'], para_dict['delta'], para_dict['rcut'])
        elif Vtype == 'bond':
            if Vstyle == 'harmonic':
                V = self._bond_harmonic(x, para_dict['kb'], para_dict['b0'])
            elif Vstyle == 'class2':
                V = self._bond_class2(x, para_dict['b0'], para_dict['k2'], para_dict['k3'], para_dict['k4'], para_dict['C'])
        elif Vtype == 'angle':
            if Vstyle == 'harmonic':
                V = self._angle_harmonic(x, para_dict['ka'], para_dict['a0'])
            elif Vstyle == 'quartic':
                V = self._angle_quartic(x, para_dict['a0'], para_dict['k2'], para_dict['k3'], para_dict['k4'], para_dict['C'])
        elif Vtype == 'dihedral':
            if Vstyle == 'harmonic':
                V = self._dihedral_harmonic(x, para_dict['kd'], para_dict['d'], para_dict['n'])
            elif Vstyle == 'multi/harmonic':
                V = self._dihedral_multiharmonic(x, para_dict['kd'])
            elif Vstyle == 'fourier':
                V = self._dihedral_fourier(x, para_dict['kd'], para_dict['n'], para_dict['d'])
        elif Vtype == 'improper':
            if Vstyle == 'harmonic':
                V = self._improper_harmonic(x, para_dict['ki'], para_dict['x0'])
        return V

    #=============================
    # pair_style lj/cut
    #=============================
    def _pair_ljcut(self, x, eps, sig, rcut):
        V = 4 * eps * ((sig / x)**12 - (sig / x)**6)
        Vcut = 4 * eps * ((sig / rcut)**12 - (sig / rcut)**6)
        mask1 = x <= rcut
        mask2 = x > rcut
        V[mask1] = V[mask1] - Vcut
        V[mask2] = 0
        return V

    def _fit_pair_ljcut(self, xall, x_fit, V_fit, weight):
        if not any(V_fit < 0):
            sig0 = x_fit[np.min(np.where(V_fit > np.max(V_fit)*0.2))]
        else:
            sig0 = x_fit[np.min(np.where(V_fit < 0))]
        eps0 = 1.0
        if sig0 > np.max(x_fit):
            sig0 = np.max(x_fit)
        elif sig0 < np.min(x_fit):
            sig0 = np.min(x_fit)
        f = lambda x, eps, sig: self._pair_ljcut(x, eps, sig, np.max(x_fit))
        popt, _ = curve_fit(f, x_fit, V_fit, p0=(eps0,sig0), bounds=([0.001,np.min(x_fit)],[np.inf,np.max(x_fit)]), sigma=1.0/weight)
        Vall = f(xall, *popt)
        self.potential['pair']['lj/cut']['eps'] = popt[0]
        self.potential['pair']['lj/cut']['sig'] = popt[1]
        self.potential['pair']['lj/cut']['rcut'] = np.max(x_fit)
        return Vall

    #=============================
    # pair_style lj96/cut
    #=============================
    def _pair_lj96cut(self, x, eps, sig, rcut):
        V = 4 * eps * ((sig / x)**9 - (sig / x)**6)
        Vcut = 4 * eps * ((sig / rcut)**9 - (sig / rcut)**6)
        mask1 = x <= rcut
        mask2 = x > rcut
        V[mask1] = V[mask1] - Vcut
        V[mask2] = 0
        return V

    def _fit_pair_lj96cut(self, xall, x_fit, V_fit, weight):
        sig0 = x_fit[np.min(np.where(V_fit < 0))]
        eps0 = 1.0
        if sig0 > np.max(x_fit):
            sig0 = np.max(x_fit)
        elif sig0 < np.min(x_fit):
            sig0 = np.min(x_fit)
        f = lambda x, eps, sig: self._pair_lj96cut(x, eps, sig, np.max(x_fit))
        popt, _ = curve_fit(f, x_fit, V_fit, p0=(eps0,sig0), bounds=([0.001,np.min(x_fit)],[np.inf,np.max(x_fit)]), sigma=1.0/weight)
        Vall = f(xall, *popt)
        self.potential['pair']['lj96/cut']['eps'] = popt[0]
        self.potential['pair']['lj96/cut']['sig'] = popt[1]
        self.potential['pair']['lj96/cut']['rcut'] = np.max(x_fit)
        return Vall

    #=============================
    # pair_style lj/expand
    #=============================
    def _pair_ljexpand(self, x, eps, sig, delta, rcut):
        V = 4 * eps * ((sig / (x - delta))**9 - (sig / (x - delta))**6)
        Vcut = 4 * eps * ((sig / (rcut + delta))**9 - (sig / (rcut + delta))**6)
        mask1 = x <= rcut + delta
        mask2 = x > rcut + delta
        V[mask1] = V[mask1] - Vcut
        V[mask2] = 0
        return V

    def _fit_pair_ljexpand(self, xall, x_fit, V_fit, weight):
        delta0 = np.min(x_fit) * 0.99
        sig0 = x_fit[np.min(np.where(V_fit < 0))] - delta0
        eps0 = 1.0
        print("sig0 = %f, delta0 = %f" % (sig0, delta0))
        f = lambda x, eps, sig, delta: self._pair_ljexpand(x, eps, sig, delta, np.max(x_fit)-delta)
        popt, _ = curve_fit(f, x_fit, V_fit, p0=(eps0,sig0,delta0), bounds=([0.001,0.0,0],[np.inf,np.max(x_fit),sig0+delta0]), sigma=1.0/weight)
        Vall = f(xall, *popt)
        self.potential['pair']['lj/expand']['eps'] = popt[0]
        self.potential['pair']['lj/expand']['sig'] = popt[1]
        self.potential['pair']['lj/expand']['delta'] = popt[2]
        self.potential['pair']['lj/expand']['rcut'] = np.max(x_fit) - popt[2]
        return Vall

    #=============================
    # bond_style harmonic
    #=============================
    def _bond_harmonic(self, x, K, x0):
        return K * (x - x0)**2

    def _fit_bond_harmonic(self, xall, x_fit, V_fit, weight):
        popt, _ = curve_fit(self._bond_harmonic, x_fit, V_fit, p0=(100,x_fit[np.argmin(V_fit)]), bounds=(0.0,np.inf), sigma=1.0/weight)
        Vall = self._bond_harmonic(xall, *popt)
        self.potential['bond']['harmonic']['kb'] = popt[0]
        self.potential['bond']['harmonic']['b0'] = popt[1]
        return Vall

    #=============================
    # bond_style class2
    #=============================
    def _bond_class2(self, x, x0, K2, K3, K4, C):
        ### C is not included in LAMMMPS
        return K2 * (x - x0)**2 + K3 * (x - x0)**3 + K4 * (x - x0)**4 + C

    def _fit_bond_class2(self, xall, x_fit, V_fit, weight):
        ssr  = {}
        ### try quadratic fit
        i = 0
        while i < 30:
            tmpf = lambda x, x0, K2: self._bond_class2(x, x0, K2, 0., 0., 0.)
            p0 = (x_fit[np.argmin(V_fit)], np.random.random()*100)
            try:
                popt, _ = curve_fit(tmpf, x_fit, V_fit, p0=p0, bounds=([0.,0.],[180.,np.inf]), sigma=1.0/weight)
            except:
                continue
            V_ = tmpf(x_fit, *popt)
            ssr_ = np.sum(((V_ - V_fit)*weight)**2)
            ssr[tuple([popt[0],popt[1],0.,0.,0.])] = ssr_
            i += 1
        ### try quartic fit
        i = 0
        while i < 30:
            p0 = (x_fit[np.argmin(V_fit)], np.random.random()*100, np.random.random()*100, np.random.random()*100, 0.0)
            try:
                popt, _ = curve_fit(self._bond_class2, x_fit, V_fit, p0=p0, bounds=([0, -np.inf, -np.inf, 0.0, -np.inf], np.inf), sigma=1.0/weight)
            except:
                continue
            V_ = self._bond_class2(x_fit, *popt)
            ssr_ = np.sum(((V_ - V_fit)*weight)**2)
            Vall_ = self._bond_class2(xall, *popt)
            ### check if right branch increase monotonically
            rmask = xall > np.max(x_fit)
            if np.sum(rmask) > 1:
                delta = Vall_[rmask][1:] - Vall_[rmask][:-1]
                if not np.all(delta > 0):
                    i += 1
                    continue
            ### check if left branch decrease monotonically
            lmask = xall < np.min(x_fit)
            if np.sum(lmask) > 1:
                delta = Vall_[lmask][1:] - Vall_[lmask][:-1]
                if not np.all(delta < 0):
                    i += 1
                    continue
            ssr[tuple(popt.tolist())] = ssr_
            i += 1

        minssr = min(ssr.values())
        for key in ssr.keys():
            if ssr[key] == minssr:
                popt = key
                break
        Vall = self._bond_class2(xall, *popt)
        Vall -= np.min(Vall)
        self.potential['bond']['class2']['b0'] = popt[0]
        self.potential['bond']['class2']['k2'] = popt[1]
        self.potential['bond']['class2']['k3'] = popt[2]
        self.potential['bond']['class2']['k4'] = popt[3]
        self.potential['bond']['class2']['C']  = popt[4]
        return Vall

    #=============================
    # angle_style harmonic
    #=============================
    def _angle_harmonic(self, x, K, x0):
        ### unit of K in LAMMPS is energy/radian^2, unit of x and x0 in LAMMPS is degree
        return (np.pi / 180)**2 * K * (x - x0)**2

    def _fit_angle_harmonic(self, xall, x_fit, V_fit, weight):
        popt, _ = curve_fit(self._angle_harmonic, x_fit, V_fit, p0=(100,x_fit[np.argmin(V_fit)]), bounds=(0.0,np.inf), sigma=1.0/weight)
        Vall = self._angle_harmonic(xall, *popt)
        self.potential['angle']['harmonic']['ka'] = popt[0]
        self.potential['angle']['harmonic']['a0'] = popt[1]
        return Vall

    #=============================
    # angle_style quartic
    #=============================
    def _angle_quartic(self, x, x0, K2, K3, K4, C):
        ### unit of K in LAMMPS is energy/radian^2, unit of x and x0 in LAMMPS is degree
        ### C is not included in LAMMMPS
        return (np.pi / 180)**2 * K2 * (x - x0)**2 + (np.pi / 180)**3 * K3 * (x - x0)**3 + (np.pi / 180)**4 * K4 * (x - x0)**4 + C

    def _fit_angle_quartic(self, xall, x_fit, V_fit, weight):
        ssr  = {}
        ### try quadratic fit
        i = 0
        while i < 30:
            tmpf = lambda x, x0, K2: self._angle_quartic(x, x0, K2, 0., 0., 0.)
            p0 = (x_fit[np.argmin(V_fit)], np.random.random()*100)
            try:
                popt, _ = curve_fit(tmpf, x_fit, V_fit, p0=p0, bounds=([0.,0.],[180.,np.inf]), sigma=1.0/weight)
            except:
                continue
            V_ = tmpf(x_fit, *popt)
            ssr_ = np.sum(((V_ - V_fit)*weight)**2)
            ssr[tuple([popt[0],popt[1],0.,0.,0.])] = ssr_
            i += 1
        ### try quartic fit
        i = 0
        while i < 30:
            p0 = (x_fit[np.argmin(V_fit)], np.random.random()*100, np.random.random()*100, np.random.random()*100, 0.0)
            try:
                popt, _ = curve_fit(self._angle_quartic, x_fit, V_fit, p0=p0, bounds=([-np.inf, -np.inf, -np.inf, 0.0, -np.inf], np.inf), sigma=1.0/weight)
            except:
                continue
            V_ = self._angle_quartic(x_fit, *popt)
            ssr_ = np.sum(((V_ - V_fit)*weight)**2)
            Vall_ = self._angle_quartic(xall, *popt)
            ### check if right branch increase monotonically
            rmask = xall > np.max(x_fit)
            if np.sum(rmask) > 1:
                delta = Vall_[rmask][1:] - Vall_[rmask][:-1]
                if not np.all(delta > 0):
                    i += 1
                    continue
            ### check if left branch decrease monotonically
            lmask = xall < np.min(x_fit)
            if np.sum(lmask) > 1:
                delta = Vall_[lmask][1:] - Vall_[lmask][:-1]
                if not np.all(delta < 0):
                    i += 1
                    continue
            ssr[tuple(popt.tolist())] = ssr_
            i += 1

        minssr = min(ssr.values())
        for key in ssr.keys():
            if ssr[key] == minssr:
                popt = key
                break
        Vall = self._angle_quartic(xall, *popt)
        Vall -= np.min(Vall)
        self.potential['angle']['quartic']['a0'] = popt[0]
        self.potential['angle']['quartic']['k2'] = popt[1]
        self.potential['angle']['quartic']['k3'] = popt[2]
        self.potential['angle']['quartic']['k4'] = popt[3]
        self.potential['angle']['quartic']['C']  = popt[4]
        return Vall

    #=============================
    # dihedral_style harmonic
    #=============================
    def _dihedral_harmonic(self, x, K, d, n):
        '''
        x in dihedral angle in degree
        '''
        x = np.pi / 180.0 * x
        return K * (1 + d * np.cos(n*x))

    def _fit_dihedral_harmonic(self, xall, x_fit, V_fit, weight):
        para = [(1, n) for n in range(1,5)] + [(-1, n) for n in range(1,5)]
        Kfit = {}
        ssr  = {}   ### sum square of residues
        for d,n in para:
            f = lambda x, K: self._dihedral_harmonic(x, K, d, n)
            popt, _ = curve_fit(f, x_fit, V_fit, p0=(0.1), bounds=(0.0,np.inf), sigma=1.0/weight)
            ssr_ = np.sum((f(x_fit, popt[0]) - V_fit)**2)
            Kfit[(d,n)] = popt[0]
            ssr[(d,n)]  = ssr_
        minssr = min(ssr.values())
        for key in ssr.keys():
            if ssr[key] == minssr:
                K = Kfit[key]
                d = key[0]
                n = key[1]
                Vall = self._dihedral_harmonic(xall, K, d, n)
                self.potential['dihedral']['harmonic']['kd'] = K
                self.potential['dihedral']['harmonic']['d'] = d
                self.potential['dihedral']['harmonic']['n'] = n
                return Vall

    #=============================
    # dihedral_style multi/harmonic
    #=============================
    def _dihedral_multiharmonic(self, x, K):
        '''
        x in dihedral angle in degree
        '''
        x = np.pi / 180.0 * x
        y = 0
        for i in range(len(K)):
            y += K[i] * np.cos(x)**(i)
        return y

    def _fit_dihedral_multiharmonic(self, xall, x_fit, V_fit, weight):
        f = lambda x, *para: self._dihedral_multiharmonic(x, para)
        popt, _ = curve_fit(f, x_fit, V_fit, p0=([1,1,1,1,1]), sigma=1.0/weight)
        Vall = self._dihedral_multiharmonic(xall, popt)
        self.potential['dihedral']['multi/harmonic']['kd'] = popt
        return Vall

    #=============================
    # dihedral_style fourier
    #=============================
    def _dihedral_fourier(self, x, K, n, d):
        '''
        x in dihedral angle in degree
        '''
        x = np.pi / 180.0 * x
        V = np.zeros(len(x))
        for i in range(len(K)):
            d_ = np.pi / 180.0 * d[i]
            V += K[i] * (1 + np.cos(n[i] * x - d_))
        return V

    def _fit_dihedral_fourier(self, xall, x_fit, V_fit, weight):
        m = len(self.potential['dihedral']['fourier']['kd'])
        n = np.arange(1,m+1)
        f = lambda x, *para: self._dihedral_fourier(x, para[:m], n, para[m:])
        ssr = []    ### sum square of residues
        popt = []
        i = 0
        while i < 30:
            K0 = (np.random.random(m) - 0.5) * 2 / m * max(V_fit)
            d0 = (np.random.random(m) - 0.5) * 2 * 180
            p0 = K0.tolist() + d0.tolist()
            try:
                popt_, _ = curve_fit(f, x_fit, V_fit, p0=p0, sigma=1.0/weight)
            except:
                i += 1
                continue
            ssr_ = np.sum((f(x_fit, *popt_) - V_fit)**2)
            ssr.append(ssr_)
            popt.append(popt_)
            i += 1

        popt = popt[np.argmin(ssr)].tolist()
        K = popt[:m]
        d = popt[m:]
        Vall = self._dihedral_fourier(xall, K, n, d)
        self.potential['dihedral']['fourier']['kd'] = K
        self.potential['dihedral']['fourier']['d'] = d
        self.potential['dihedral']['fourier']['n'] = n.astype(int).tolist()
        return Vall

    #=============================
    # improper_style harmonic
    #=============================
    def _improper_harmonic(self, x, K, x0):
        ### unit of K in LAMMPS is energy/radian^2, unit of x and x0 in LAMMPS is degree
        return (np.pi / 180)**2 * K * (x - x0)**2

    def _fit_improper_harmonic(self, xall, x_fit, V_fit, weight):
        popt, _ = curve_fit(self._improper_harmonic, x_fit, V_fit, p0=(100,x_fit[np.argmin(V_fit)]), bounds=(0.0,np.inf), sigma=1.0/weight)
        Vall = self._improper_harmonic(xall, *popt)
        self.potential['improper']['harmonic']['ki'] = popt[0]
        self.potential['improper']['harmonic']['x0'] = popt[1]
        return Vall


