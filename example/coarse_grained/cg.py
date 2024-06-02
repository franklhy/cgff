import os
import numpy as np
from cgff import cg
from plato import data

change_cgtype = {1:3, 2:3}    ### the type of cg bead to be changed in GBCG
aapath="../atomistic/"
CGlevel = 2

aa_data = os.path.join(aapath, "atomistic.data")
aa_dump = os.path.join(aapath, "atomistic.traj")

d = data()
d.read(aa_data)
timestep,natoms,box,props,atoms = d.snapshot()
bonds,angles,dihedrals,impropers = d.topology()
ID,MOL = d.map("id", "mol")
new_atoms = atoms[atoms[:,MOL] == 1]
napc = len(new_atoms)
allid = new_atoms[:,ID]
bmask1 = np.in1d(bonds[:,2], allid)
bmask2 = np.in1d(bonds[:,3], allid)
new_bonds = bonds[bmask1 * bmask2]
d.changeatoms(new_atoms, props)
d.changebonds(new_bonds)
d.write_data("poly.data", "full")

mapfile = open("GBCG/map_files/iter.%d.map" % CGlevel, "r")
cgid = []
cgty = []
cgm = []
cgq = []
aidtocgid = {}
for line in mapfile:
    words = line.strip().split()
    cgid_ = int(words[0])
    cgty_ = int(words[1])
    if cgty_ in change_cgtype.keys():
        cgty_ = change_cgtype[cgty_]
    cgm_ = float(words[2])
    cgq_ = float(words[3])
    aid = []
    for word in words[4:]:
        if int(word) > napc:
          break
        aid.append(int(word))
    if len(aid) == 0:
        break
    cgid.append(cgid_)
    cgty.append(cgty_)
    cgm.append(cgm_)
    cgq.append(cgq_)
    for aid_ in aid:
        aidtocgid[aid_] = cgid_

uniq_type = np.unique(cgty)
change_type = {uniq_type[i]:i+1 for i in range(len(uniq_type))}
for i in range(len(cgty)):
    cgty[i] = change_type[cgty[i]]
print(change_type)

f = open("poly.cgmap", "w")
f.write("#atomid cgid\n")
for i in range(len(aidtocgid.keys())):
    f.write("%-8d%-8d\n" % (i+1, aidtocgid[i+1]))
f.close()

f = open("poly.cgtype", "w")
f.write("#cg_id  cg_type\n")
for i in range(len(cgty)):
    f.write("%-8d%-8d\n" % (cgid[i], cgty[i]))
f.close()

f = open("poly.lmp2cgtype.map", "w")
f.write("#lammps type\n")
for i in range(len(uniq_type)):
    f.write("%-8d%-8s\n" % (i+1, "CG%d" % (i+1)))
f.close()

aa2cg = cg.atomistic2cg()
aa2cg.main("mol", aa_data, aa_dump, ["poly.data"], ["poly.cgmap"], ["poly.cgtype"], ["poly.lmp2cgtype.map"], "output")

#dist = cg.distribution()
