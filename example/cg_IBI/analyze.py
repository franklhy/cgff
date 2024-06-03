import os
import matplotlib.pyplot as plt
import cgff.IBI as IBI

path = "output/"
#steps = 30

tool = IBI.analyze(run_path=path)

fig1, ax1 = tool.plot_rmsd()
fig2, ax2 = tool.plot_distribution(Vtype="pair")
fig3, ax3 = tool.plot_potential(Vtype="pair")
fig4, ax4 = tool.plot_distribution(Vtype="bond")
fig5, ax5 = tool.plot_potential(Vtype="bond")
fig6, ax6 = tool.plot_distribution(Vtype="angle")
fig7, ax7 = tool.plot_potential(Vtype="angle")
fig8, ax8 = tool.plot_distribution(Vtype="dihedral")
fig9, ax9 = tool.plot_potential(Vtype="dihedral")
fig10, ax10 = tool.plot_pressure()
'''
fig_p, ax_p = plt.subplots(1)
for i in range(steps+1):
    if not os.path.exists(path + "%d/cg.data" % i):
        continue
    file = open(path + "%d/pressure.txt" % i)
    file.readline()
    file.readline()
    p = float(file.readline().split()[1])
    print(i,p)
    ax_p.plot(i,p,'ro')
'''
plt.show(block=True)
