import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

class color:
    def __init__(self, vmin, vmax):
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalarmap = cm.ScalarMappable(norm=cnorm, cmap="rainbow")

    def get_color(self, value):
        return self.scalarmap.to_rgba(value)

class analyze:
    def __init__(self, run_path=".", rmsd_filename="IBI.rmsd"):
        if run_path[0] != '/':
            self.run_path = os.getcwd() + "/" + run_path + "/"
        else:
            self.run_path = run_path
        self.rmsd_filename = rmsd_filename
        ### keep handler of all matplotlib radio button widget, so they will not stop functioning
        self.radio_distribution = []
        self.radio_potential = []
        self.radio_potential_vs_nosmooth = []

    def _is_number(self, s):
        """
        If a string is a number
        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    def plot_rmsd(self):
        """
        Plot root-mean-square ddeviation of different distributions
        """
        marker = {"pair": 'o', 'bond': '^', 'angle': 'v', 'dihedral':'s', 'improper': '*'}
        steps = [int(s) for s in os.listdir(self.run_path) if os.path.isdir(self.run_path + s) and self._is_number(s)]
        steps.sort()
        rmsd = {}
        for i in steps:
            filename = self.run_path + "%d/%s" % (i, self.rmsd_filename)
            if not os.path.exists(filename):
                continue
            with open(filename) as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        name, tmp = line.split()
                        name_rmsd = float(tmp)
                        if i == 0:
                            rmsd[name] = [name_rmsd]
                        else:
                            rmsd[name].append(name_rmsd)

        fig, ax = plt.subplots(1)
        for key in rmsd.keys():
            ax.plot(rmsd[key], ls='-', marker=marker[key.split('_')[0]], label=key)
        ax.legend(loc='best')

        return fig, ax

    def plot_distribution(self, Vtype="pair"):
        """
        Plot distribution.
        Vtpye should be chosen from: "pair", "bond", "angle", "dihedral" or "improper"
        """
        target_distribution = {}
        target_path = self.run_path + "target/distribution/"
        targets = [s for s in sorted(os.listdir(target_path)) if os.path.isfile(target_path + s)]
        for target in targets:
            key = target.split(".")[0]
            words = key.split("_")
            if Vtype == "pair" and words[0] == "rdf":
                target_distribution[key] = np.loadtxt(target_path + target)
            elif Vtype == "bond" and words[0] == "bonddist":
                target_distribution[key] = np.loadtxt(target_path + target)
            elif Vtype == "angle" and words[0] == "angledist":
                target_distribution[key] = np.loadtxt(target_path + target)
            elif Vtype == "dihedral" and words[0] == "dihedraldist":
                target_distribution[key] = np.loadtxt(target_path + target)
            elif Vtype == "improper" and words[0] == "improperdist":
                target_distribution[key] = np.loadtxt(target_path + target)
        
        step_distribution = {}
        for key in target_distribution.keys():
            step_distribution[key] = {}
        steps = [int(s) for s in os.listdir(self.run_path) if os.path.isdir(self.run_path + s) and self._is_number(s)]
        steps.sort()
        for i in steps:
            for key in step_distribution.keys():
                filename = self.run_path + "%d/distribution/%s.txt" % (i, key)
                if not os.path.exists(filename):
                    continue
                step_distribution[key][i] = np.loadtxt(filename)

        colors = color(min(steps), max(steps))
        fig, ax = plt.subplots(1)
        ax.set_label('plot')
        fig.suptitle("Distribution")
        l = {}
        key = list(target_distribution.keys())[0]
        l['target'], = ax.plot(target_distribution[key][:,0], target_distribution[key][:,1], 'k-', zorder=1000, label='target')
        for step in step_distribution[key].keys():
            l[step], = ax.plot(step_distribution[key][step][:,0], step_distribution[key][step][:,1], \
                ls='-', marker='.', color=colors.get_color(step), label=step, zorder=int(step))
        fig.colorbar(colors.scalarmap, ax=ax)

        ### radio button
        axcolor = 'lightgoldenrodyellow'
        rax = fig.add_axes([0.05, 0.2, 0.15, 0.55], facecolor=axcolor)
        radio_distribution = RadioButtons(rax, [key for key in target_distribution.keys()])
        
        def _plot_distribution(label):
            l['target'].set_xdata(target_distribution[label][:,0])
            l['target'].set_ydata(target_distribution[label][:,1])
            for step in step_distribution[label].keys():
                l[step].set_xdata(step_distribution[label][step][:,0])
                l[step].set_ydata(step_distribution[label][step][:,1])
            fig.canvas.draw_idle()
        
        radio_distribution.on_clicked(_plot_distribution)
        self.radio_distribution.append(radio_distribution)
        
        ### highlight one line when hover on colorbar
        def _on_hover(event):
            if event.inaxes != None:
                if event.inaxes.get_label() == '<colorbar>':
                    infig = event.inaxes.get_figure()
                    for ax in infig.get_axes():
                        if ax.get_label() == 'plot':
                            break
                    stepid = "%d" % np.rint(event.ydata)
                    infig.canvas.toolbar.set_message(stepid)
                    for line in ax.lines:
                        if line.get_label() == stepid:
                            line.set_marker('.')
                            line.set_linewidth(3.0)
                            line.set_zorder(999)
                        elif line.get_label() == 'target':
                            pass
                        else:
                            line.set_marker('None')
                            line.set_linewidth(0.5)
                            line.set_zorder(int(line.get_label()))
                    infig.canvas.draw_idle()

        def _leave_axes(event):
            for ax in event.canvas.figure.get_axes():
                if ax.get_label() == 'plot':
                    break
            for line in ax.lines:
                if line.get_label() != 'target':
                    line.set_marker('.')
                    line.set_linewidth(1.0)
                    line.set_zorder(int(line.get_label()))
            event.canvas.figure.canvas.draw_idle()

        fig.canvas.callbacks.connect('motion_notify_event', _on_hover)
        fig.canvas.callbacks.connect('axes_leave_event', _leave_axes)

        return fig, ax

    def plot_potential(self, Vtype="pair"):
        """
        Plot tabulated potential.
        Vtpye should be chosen from: "pair", "bond", "angle", "dihedral" or "improper"
        """
        target_bi_potential = {}    ### boltzmann inversion of the target distribution
        target_path = self.run_path + "target/potential/"
        potentials = [s for s in sorted(os.listdir(target_path)) if os.path.isfile(target_path + s)]
        for potential in potentials:
            key = potential.split(".")[0]
            words = key.split("_")
            filename = target_path + "%s.txt" % key
            if Vtype == "pair" and words[0] == "pair" and words[1] == "potential":
                target_bi_potential[key] = np.loadtxt(filename, skiprows=4, usecols=(1,2))
            elif Vtype == "bond" and words[0] == "bond" and words[1] == "potential":
                target_bi_potential[key] = np.loadtxt(filename, skiprows=4, usecols=(1,2))
            elif Vtype == "angle" and words[0] == "angle" and words[1] == "potential":
                target_bi_potential[key] = np.loadtxt(filename, skiprows=4, usecols=(1,2))
            elif Vtype == "dihedral" and words[0] == "dihedral" and words[1] == "potential":
                target_bi_potential[key] = np.loadtxt(filename, skiprows=4, usecols=(1,2))
            elif Vtype == "improper" and words[0] == "improper" and words[1] == "potential":
                target_bi_potential[key] = np.loadtxt(filename, skiprows=4, usecols=(1,2))

        step_potential = {}
        for key in target_bi_potential.keys():
            step_potential[key] = {}
        steps = [int(s) for s in os.listdir(self.run_path) if os.path.isdir(self.run_path + s) and self._is_number(s)]
        steps.sort()
        for i in steps:
            for key in step_potential.keys():
                filename = self.run_path + "%d/potential/%s.txt" % (i, key)
                if not os.path.exists(filename):
                    continue
                step_potential[key][i] = np.loadtxt(filename, skiprows=4, usecols=(1,2))

        colors = color(min(steps), max(steps))
        fig, ax = plt.subplots(1)
        ax.set_label('plot')
        fig.suptitle("Input potential")
        l = {}
        key = list(target_bi_potential.keys())[0]
        l['target'], = ax.plot(target_bi_potential[key][:,0], target_bi_potential[key][:,1], 'k-', zorder=1000, label='target')
        if list(step_potential.keys()):
            key = list(step_potential.keys())[0]
            for step in step_potential[key].keys():
                l[step], = ax.plot(step_potential[key][step][:,0], step_potential[key][step][:,1], \
                    ls='-', marker='.', color=colors.get_color(step), label=step, zorder=int(step))
            fig.colorbar(colors.scalarmap, ax=ax)
            
            ### radio button
            axcolor = 'lightgoldenrodyellow'
            rax = fig.add_axes([0.05, 0.2, 0.15, 0.55], facecolor=axcolor)
            radio_potential = RadioButtons(rax, [key for key in step_potential.keys()])

            def _plot_potential(label):
                l['target'].set_xdata(target_bi_potential[label][:,0])
                l['target'].set_ydata(target_bi_potential[label][:,1])
                for step in step_potential[label].keys():
                    l[step].set_xdata(step_potential[label][step][:,0])
                    l[step].set_ydata(step_potential[label][step][:,1])
                fig.canvas.draw_idle()
            
            radio_potential.on_clicked(_plot_potential)
            self.radio_potential.append(radio_potential)

            ### highlight one line when hover on colorbar
            def _on_hover(event):
                if event.inaxes != None:
                    if event.inaxes.get_label() == '<colorbar>':
                        infig = event.inaxes.get_figure()
                        for ax in infig.get_axes():
                            if ax.get_label() == 'plot':
                                break
                        stepid = "%d" % np.rint(event.ydata)
                        infig.canvas.toolbar.set_message(stepid)
                        for line in ax.lines:
                            if line.get_label() == stepid:
                                line.set_marker('.')
                                line.set_linewidth(3.0)
                                line.set_zorder(999)
                            elif line.get_label() == 'target':
                                pass
                            else:
                                line.set_marker('None')
                                line.set_linewidth(0.5)
                                line.set_zorder(int(line.get_label()))
                        infig.canvas.draw_idle()

            def _leave_axes(event):
                for ax in event.canvas.figure.get_axes():
                    if ax.get_label() == 'plot':
                        break
                for line in ax.lines:
                    if line.get_label() != 'target':
                        line.set_marker('.')
                        line.set_linewidth(1.0)
                        line.set_zorder(int(line.get_label()))
                event.canvas.figure.canvas.draw_idle()

            fig.canvas.callbacks.connect('motion_notify_event', _on_hover)
            fig.canvas.callbacks.connect('axes_leave_event', _leave_axes)
        else:
            ax.text(0.0, 0.5, "No tabulated potential!\n" + \
                "The corresponding potential is fitted to an analytical form.\n" +
                "See the LAMMPS input script for details.")

        return fig, ax

    '''
    def plot_potential_vs_nosmooth(self, Vtype="pair"):
        """
        Plot tabulated potential and compare to that not being smoothed.
        Vtpye should be chosen from: "pair", "bond", "angle", "dihedral" or "improper"
        """
        step_potential = {}
        step_potential_nosmooth = {}
        tmp_path = self.run_path + "0/potential/"
        potentials = [s for s in sorted(os.listdir(tmp_path)) if os.path.isfile(tmp_path + s)]
        for potential in potentials:
            key = potential.split(".")[0]
            words = key.split("_")
            if Vtype == "pair" and words[0] == "pair" and words[1] == "potential":
                step_potential[key] = {}
                step_potential_nosmooth[key] = {}
            elif Vtype == "bond" and words[0] == "bond" and words[1] == "potential":
                step_potential[key] = {}
                step_potential_nosmooth[key] = {}
            elif Vtype == "angle" and words[0] == "angle" and words[1] == "potential":
                step_potential[key] = {}
                step_potential_nosmooth[key] = {}
            elif Vtype == "dihedral" and words[0] == "dihedral" and words[1] == "potential":
                step_potential[key] = {}
                step_potential_nosmooth[key] = {}
            elif Vtype == "improper" and words[0] == "improper" and words[1] == "potential":
                step_potential[key] = {}
                step_potential_nosmooth[key] = {}
        
        steps = [int(s) for s in os.listdir(self.run_path) if os.path.isdir(self.run_path + s) and self._is_number(s)]
        steps.sort()
        for i in steps:
            for key in step_potential.keys():
                ### smoothed
                filename = self.run_path + "%d/potential/%s.txt" % (i, key)
                if not os.path.exists(filename):
                    continue
                step_potential[key][i] = np.loadtxt(filename, skiprows=4, usecols=(1,2))
                ### not smoothed
                words = key.split("_")
                tmpstring1 = "%s_%s_nosmooth" % (words[0], words[1])
                tmpstring2 = ""
                for j in range(2, len(words)):
                    tmpstring2 += "_%s" % words[j]
                tmpstring = tmpstring1 + tmpstring2
                filename = self.run_path + "%d/potential/%s.txt" % (i, tmpstring)
                if not os.path.exists(filename):
                    continue
                step_potential_nosmooth[key][i] = np.loadtxt(filename, skiprows=4, usecols=(1,2))

        colors = color(min(steps), max(steps))
        fig, ax = plt.subplots(1)
        fig.suptitle("input potential (smoothed, line) vs unsmoothed potential (dot)")
        l = {}
        l_nosmooth = {}
        if list(step_potential.keys()):
            key = list(step_potential.keys())[0]
            for step in step_potential[key].keys():
                l[step], = ax.plot(step_potential[key][step][:,0], step_potential[key][step][:,1], \
                    ls='-', color=colors.get_color(step))
            for step in step_potential_nosmooth[key].keys():
                l_nosmooth[step], = ax.plot(step_potential_nosmooth[key][step][:,0], step_potential_nosmooth[key][step][:,1], \
                    ls='none', marker='.', color=colors.get_color(step))
            fig.colorbar(colors.scalarmap, ax=ax)
                
            axcolor = 'lightgoldenrodyellow'
            rax = fig.add_axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
            radio_potential_vs_nosmooth = RadioButtons(rax, [key for key in step_potential.keys()])

            def _plot_potential_vs_nosmooth(label):
                for step in step_potential[label].keys():
                    l[step].set_xdata(step_potential[label][step][:,0])
                    l[step].set_ydata(step_potential[label][step][:,1])
                for step in step_potential_nosmooth[label].keys():
                    l_nosmooth[step].set_xdata(step_potential_nosmooth[label][step][:,0])
                    l_nosmooth[step].set_ydata(step_potential_nosmooth[label][step][:,1])
                fig.canvas.draw_idle()
            
            radio_potential_vs_nosmooth.on_clicked(_plot_potential_vs_nosmooth)
            self.radio_potential_vs_nosmooth.append(radio_potential_vs_nosmooth)
        else:
            ax.text(0.0, 0.5, "No tabulated potential!\n" + \
                "The corresponding potential is fitted to an analytical form.\n" +
                "See the LAMMPS input script for details.")
    '''