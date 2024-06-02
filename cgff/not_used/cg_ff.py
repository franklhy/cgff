import os
import sys
import copy
import shutil
import importlib
from .ff_database import ff_database

class cg_ff:
    def __init__(self):
        self.ff_db = None
        self.ff_db_path = None

    def _read_typemap(self, typemapfile):
        id_to_name = ['']       ## id_to_name[0] = '' is a place holder
        name_to_id = {}
        if typemapfile is not None:
            try:
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
                                name_to_id[name] = ti
                            else:
                                raise RuntimeError("LAMMPS type in type mapping file should be continuous integer" + \
                                    "staring from 1.")
            except:
                raise RuntimeError("Error reading typemapfile %s" % typemapfile)

        return id_to_name, name_to_id

    def _assign_lammps_atom_type(self, atom, id_to_name=[''], name_to_id={}):
        '''
        Assign LAMMPS atom type to each atom in the chain and build a atom type map for LAMMPS. 

        Input:
            atom: a dictionary of atoms, key:int, value:instance of class ATOM
                A dictionary of atoms. Key: atom id. Value: the atom (class ATOM)

        Return:
            typ: list
                LAMMPS atom type. typ[atom id] = LAMMPS atom type

            id_to_name: list
                Atom type map. If i is the atom type in LAMMPS, then id_to_name[i] = cg atom type (string)

            miss_type: list
                A list of missing atom type.
        '''
        miss_typ   = []
        typ        = [0]        ### typ[0] = 0 is a place holder
        #for atom_ in atom.values():
        for i in range(1, 1 + len(atom)):
            atom_ = atom[i]
            t = atom_.type
            if t in name_to_id.keys():
                typ.append(name_to_id[t])
            elif t in self.ff_db.atm_types.keys():
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
            else:
                print("Missing atom type: %s" % t)
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
                miss_typ.append(t)
                self.ff_db.add_dummy_type("atom", t)
        return typ, id_to_name, miss_typ

    def _assign_lammps_bond_type(self, atom, bond, id_to_name=[''], name_to_id={}):
        miss_typ   = []
        typ        = []
        for (i,j) in bond:
            ti = atom[i].type
            tj = atom[j].type
            t  = (ti, tj)
            if t[::-1] in name_to_id.keys() or t[::-1] in self.ff_db.bon_types.keys():
                t = t[::-1]
            if t in name_to_id.keys():
                typ.append(name_to_id[t])
            elif t in self.ff_db.bon_types.keys():
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
            else:
                print("Missing bond type: (%s, %s)." % (ti, tj))
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
                miss_typ.append(t)
                self.ff_db.add_dummy_type("bond", t)
        return typ, id_to_name, miss_typ

    def _assign_lammps_angle_type(self, atom, angle, id_to_name=[''], name_to_id={}):
        miss_typ   = []
        typ        = []
        for (i,j,k) in angle:
            ti = atom[i].type
            tj = atom[j].type
            tk = atom[k].type
            t  = (ti, tj, tk)
            if t[::-1] in name_to_id.keys() or t[::-1] in self.ff_db.ang_types.keys():
                t = t[::-1]
            if t in name_to_id.keys():
                typ.append(name_to_id[t])
            elif t in self.ff_db.ang_types.keys():
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
            else:
                print("Missing angle type: (%s, %s, %s)." % (ti, tj, tk))
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
                miss_typ.append(t)
                self.ff_db.add_dummy_type("angle", t)
        return typ, id_to_name, miss_typ

    def _assign_lammps_dihedral_type(self, atom, dihedral, id_to_name=[''], name_to_id={}):
        miss_typ   = []
        typ        = []
        for (i,j,k,l) in dihedral:
            ti = atom[i].type
            tj = atom[j].type
            tk = atom[k].type
            tl = atom[l].type
            t  = (ti, tj, tk, tl)
            if t[::-1] in name_to_id.keys() or t[::-1] in self.ff_db.dih_types.keys():
                t = t[::-1]
            if t in name_to_id.keys():
                typ.append(name_to_id[t])
            elif t in self.ff_db.dih_types.keys():
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
            else:
                print("Missing dihedral type: (%s, %s, %s, %s)." % (ti, tj, tk, tl))
                name_to_id[t] = len(name_to_id) + 1
                id_to_name.append(t)
                typ.append(name_to_id[t])
                miss_typ.append(t)
                self.ff_db.add_dummy_type("dihedral", t)
        return typ, id_to_name, miss_typ

    def _table_potential_style_string(self, potential_dict):
        table_Nmax = 0
        for key in potential_dict.keys():
            para = potential_dict[key]
            table_f = open(self.ff_db_path + "/" + para['file'], "r")
            for line in table_f:
                if line[0] == 'N':
                    Nfile = int(line.split()[1])
                    if Nfile > table_Nmax:
                        table_Nmax = Nfile
                    break
            table_f.close()
        style_str = "table linear %d" % table_Nmax
        return style_str

    def _table_potential_coeff_string(self, Vstyle, Vtype, outputfolder, filename, keyword):
        if outputfolder[-1] == '/':
            table_folder = outputfolder + "/potential"
        else:
            table_folder = outputfolder + "potential"
        if not os.path.exists(table_folder):
            os.mkdir(table_folder)
        table_file = self.ff_db_path + "/" + filename
        #_, filename = os.path.split(table_file)
        if Vstyle == "pair":
            newfilename = "%s_potential_%d_%d.txt" % (Vstyle, Vtype[0], Vtype[1])
        else:
            newfilename = "%s_potential_%d.txt" % (Vstyle, Vtype)
        fold = open(table_file, "r")
        fnew = open(table_folder + "/" + newfilename, "w")
        fnew.write("# copy from %s\n" % (table_file))
        for line in fold:
            fnew.write(line)
        fold.close()
        fnew.close()
        #shutil.copy2(table_file, table_folder + "/" + newfilename)
        if Vstyle == "pair":
            coeff_str = "%s_coeff\t%d %d %s %s" % (Vstyle, Vtype[0], Vtype[1], "potential/" + newfilename, keyword)
        else:
            coeff_str = "%s_coeff\t%d %s %s" % (Vstyle, Vtype, "potential/" + newfilename, keyword)
        return coeff_str

    def write_lammps(self, cell, forcefield_database, md_setup_file, datafile, settingfile, lammpsscript, typemap, outputfolder, \
        use_atom_typemap=None, use_bond_typemap=None, use_angle_typemap=None, use_dihedral_typemap=None):
        '''
        cell: an instance of class CELL
            The CELL instance including all atoms to be included in this all-atom simulation.

        forcefield_database: string
            The file name of the cg force field database.

        datafile: string
            The name of the new lammps datafile.

        settingfile: string
            The name of the new lammps script defining the force field.

        lammpsscript: string
            The name of the new lammps script for the all-atom simulation.

        typemap: string
            The name prefix of new type mapping files. Five files, named *.atom, *.bond, *.angle, and *.dihedral
            (where * is the name prefix), will be generated. Each file lists the lammps type id and the corresponding 
            type name in cg force field.
        '''
        ### try to read typemap
        use_ttym = self._read_typemap(use_atom_typemap)
        use_btym = self._read_typemap(use_bond_typemap)
        use_atym = self._read_typemap(use_angle_typemap)
        use_dtym = self._read_typemap(use_dihedral_typemap)

        self.ff_db = ff_database()
        ### load cg force field database
        self.ff_db_path, name = os.path.split(forcefield_database)
        if self.ff_db_path == '':
            self.ff_db_path = './'
        sys.path.append(os.path.abspath(self.ff_db_path))
        ff_db = importlib.import_module(name[:-3])
        sys.path.pop()
        self.ff_db.atm_types.update(ff_db.atm_types)
        self.ff_db.par_types.update(ff_db.par_types)
        self.ff_db.bon_types.update(ff_db.bon_types)
        self.ff_db.ang_types.update(ff_db.ang_types)
        self.ff_db.dih_types.update(ff_db.dih_types)

        atom = cell.atom
        bond = cell.build_bond(atom)
        angl = cell.build_angle(atom, bond)
        dihe = cell.build_dihedral(atom, angl)
        print("Building cg atom type mapping")
        ttyp, ttym, mstt = self._assign_lammps_atom_type(atom, use_ttym[0], use_ttym[1])
        print("Building cg bond type mapping")
        btyp, btym, msbt = self._assign_lammps_bond_type(atom, bond, use_btym[0], use_btym[1])
        print("Building cg angle type mapping")
        atyp, atym, msat = self._assign_lammps_angle_type(atom, angl, use_atym[0], use_atym[1])
        print("Building cg dihedral type mapping")
        dtyp, dtym, msdt = self._assign_lammps_dihedral_type(atom, dihe, use_dtym[0], use_dtym[1])

        ### missing types
        if (len(mstt) + len(msbt) + len(msat) + len(msdt)) != 0:
            f = open(outputfolder + "/" + "missing_type.txt", "w")
            f.write("Missing atom types:\n")
            for t in mstt:
                f.write("%-8s\n" % t)
            f.write("\n")
            f.write("Missing bond types:\n")
            for t in msbt:
                f.write("%-8s %-8s\n" % (t[0], t[1]))
            f.write("\n")
            f.write("Missing angle types:\n")
            for t in msat:
                f.write("%-8s %-8s %-8s\n" % (t[0], t[1], t[2]))
            f.write("\n")
            f.write("Missing dihedral types:\n")
            for t in msdt:
                f.write("%-8s %-8s %-8s %-8s\n" % (t[0], t[1], t[2], t[3]))
            f.write("\n")
            print("Missing atom/bond/angle/dihedral types. See file \"missing_type.txt\" for more information.")
        else:
            print("Great! No missing types!")
  
        ### generate the cg atom type mapping file
        f = open(outputfolder + "/" + typemap + ".atom", "w")
        f.write("# Generated by cgff\n")
        for i in range(1, len(ttym)):
            f.write("%d\t%s\n" % (i, ttym[i]))
        f.write("\n")
        f.close()

        ### generate the cg bond type mapping file
        f = open(outputfolder + "/" + typemap + ".bond", "w")
        f.write("# Generated by cgff\n")
        for i in range(1, len(btym)):
            f.write("%d\t%-8s %-8s\n" % (i, *btym[i]))
        f.write("\n")
        f.close()

        ### generate the cg angle type mapping file
        f = open(outputfolder + "/" + typemap + ".angle", "w")
        f.write("# Generated by cgff\n")
        for i in range(1, len(atym)):
            f.write("%d\t%-8s %-8s %-8s\n" % (i, *atym[i]))
        f.write("\n")
        f.close()

        ### generate the cg dihedral type mapping file
        f = open(outputfolder + "/" + typemap + ".dihedral", "w")
        f.write("# Generated by cfgg\n")
        for i in range(1, len(dtym)):
            f.write("%d\t%-8s %-8s %-8s %-8s\n" % (i, *dtym[i]))
        f.write("\n")
        f.close()

        print("Writing lammps data file and force field...")

        ### generate LAMMPS data file
        f = open(outputfolder + "/" + datafile, "w")
        f.write("# Generated by cgff\n")
        f.write("%d atoms\n" % len(atom))
        f.write("%d bonds\n" % len(bond))
        f.write("%d angles\n" % len(angl))
        f.write("%d dihedrals\n" % len(dihe))
        f.write("%d atom types\n" % (len(ttym) - 1))
        f.write("%d bond types\n" % (len(btym) - 1))
        f.write("%d angle types\n" % (len(atym) - 1))
        f.write("%d dihedral types\n" % (len(dtym) - 1))
        f.write("\n")
        f.write("%f %f xlo xhi\n" % (cell.box[0], cell.box[3]))
        f.write("%f %f ylo yhi\n" % (cell.box[1], cell.box[4]))
        f.write("%f %f zlo zhi\n" % (cell.box[2], cell.box[5]))
        f.write("\n")
        f.write("Masses\n\n")
        for i in range(1, len(ttym)):
            f.write("%d %f\n" % (i, self.ff_db.atm_types[ttym[i]]['m']))
        f.write("\n")
        f.write("Atoms # full\n\n")
        for i in range(1, len(atom)+1):
            f.write("%d %d %d %f %f %f %f 0 0 0\n" % (atom[i].aid, atom[i].mid, ttyp[i], atom[i].chrg, atom[i].crd[0], atom[i].crd[1], atom[i].crd[2]))
        f.write("\n")
        if len(bond) > 0:
            f.write("Bonds\n\n")
            for i in range(len(bond)):
                f.write("%d %d %d %d\n" % (i+1, btyp[i], *bond[i]))
            f.write("\n")
        if len(angl) > 0:
            f.write("Angles\n\n")
            for i in range(len(angl)):
                f.write("%d %d %d %d %d\n" % (i+1, atyp[i], *angl[i]))
            f.write("\n")
        if len(dihe) > 0:
            f.write("Dihedrals\n\n")
            for i in range(len(dihe)):
                f.write("%d %d %d %d %d %d\n" % (i+1, dtyp[i], *dihe[i]))
            f.write("\n")
        f.close()

        ### write force field setting file
        f = open(outputfolder + "/" + settingfile, "w")
        f.write("# CG Force-field settings file generated by cgff\n")
        ### pairwise interaction
        f.write("\n########PAIRS#########\n")
        psty_str = None
        psty = set([self.ff_db.par_types[par]['style'] for par in self.ff_db.par_types.keys()])
        psty = list(psty)
        if len(psty) > 1:
            raise NotImplementedError("No support for hybrid potential.")
        if psty[0] == 'table':
            psty_str = self._table_potential_style_string(self.ff_db.par_types)
        elif psty[0] == 'lj/cut':
            ljcut_rcutmax = 0
            for par in self.ff_db.par_types.keys():
                if self.ff_db.par_types[par]['rcut'] > ljcut_rcutmax:
                    ljcut_rcutmax = self.ff_db.par_types[par]['rcut']
            psty_str = "lj/cut %g" % ljcut_rcutmax
        elif psty[0] == 'lj96/cut':
            lj96cut_rcutmax = 0
            for par in self.ff_db.par_types.keys():
                if self.ff_db.par_types[par]['rcut'] > lj96cut_rcutmax:
                    lj96cut_rcutmax = self.ff_db.par_types[par]['rcut']
            psty_str = "lj96/cut %g" % lj96cut_rcutmax
        else:
            raise ValueError("Invalid pair style.")
        f.write("#pair_style\t%s\n" % psty_str)
        for i in range(1, len(ttym)):
            for j in range(i, len(ttym)):
                if (ttym[i], ttym[j]) in self.ff_db.par_types.keys():
                    para = self.ff_db.par_types[(ttym[i], ttym[j])]
                elif (ttym[j], ttym[i]) in self.ff_db.par_types.keys():
                    para = self.ff_db.par_types[(ttym[j], ttym[i])]
                else:
                    raise RuntimeError()    ### TODO: add _match_pair_type
                if para['style'] == 'table':
                    tmps =self._table_potential_coeff_string("pair", (i, j), outputfolder, para['file'], para['keyw'])
                    f.write("%s" % tmps)
                elif para['style'] == 'lj/cut':
                    f.write("pair_coeff\t%d %d %g %g %g" % (i, j, para['eps'], para['sig'], para['rcut']))
                elif para['style'] == 'lj96/cut':
                    f.write("pair_coeff\t%d %d %g %g %g" % (i, j, para['eps'], para['sig'], para['rcut']))
                f.write("\t\t#%8s %8s\n" % (ttym[i], ttym[j]))
        f.write("\n")

        ### bond force field
        f.write("\n########BONDS#########\n")
        bsty = set([self.ff_db.bon_types[bon]['style'] for bon in btym[1:]])
        bsty = list(bsty)
        if len(bsty) > 1:
            raise NotImplementedError("No support for hybrid potential.")
        if bsty[0] == 'table':
            bsty_str = self._table_potential_style_string(self.ff_db.bon_types)
        elif bsty[0] == 'none':
            bsty_str = "none"
        elif bsty[0] == 'harmonic':
            bsty_str = "harmonic"
        elif bsty[0] == 'class2':
            bsty_str = "class2"
        else:
            raise ValueError("Invalid bond style.")
        f.write("#bond_style\t%s\n" % bsty_str)
        for i in range(1, len(btym)):
            para = self.ff_db.bon_types[btym[i]]
            if para['style'] == 'table':
                tmps = self._table_potential_coeff_string("bond", i, outputfolder, para['file'], para['keyw'])
                f.write("%s" % tmps)
            elif para['style'] == 'none':
                pass
            elif para['style'] == 'harmonic':
                f.write("bond_coeff\t%3d %g %g" % (i, para['kb'], para['b0']))
            elif para['style'] == 'class2':
                f.write("bond_coeff\t%3d %g %g %g %g" % (i, para['b0'], para['k2'], para['k3'], para['k4']))
            f.write("\t\t#%8s %8s\n" % (btym[i][0], btym[i][1]))

        ## angle force field
        f.write("\n########ANGLES#########\n")
        asty = set([self.ff_db.ang_types[ang]['style'] for ang in atym[1:]])
        asty = list(asty)
        if len(asty) > 1:
            raise NotImplementedError("No support for hybrid potential.")
        if asty[0] == 'table':
            asty_str = self._table_potential_style_string(self.ff_db.ang_types)
        elif asty[0] == 'none':
            asty_str = "none"
        elif asty[0] == 'harmonic':
            asty_str = "harmonic"
        elif asty[0] == 'quartic':
            asty_str = "quartic"
        else:
            raise ValueError("Invalid angle style.")
        f.write("#angle_style\t%s\n" % asty_str)
        for i in range(1, len(atym)):
            para = self.ff_db.ang_types[atym[i]]
            if para['style'] == 'table':
                tmps = self._table_potential_coeff_string("angle", i, outputfolder, para['file'], para['keyw'])
                f.write("%s" % tmps)
            elif para['style'] == 'none':
                pass
            elif para['style'] == 'harmonic':
                f.write("angle_coeff\t%d %g %g" % (i, para['ka'], para['a0']))
            elif para['style'] == 'quartic':
                f.write("angle_coeff\t%d %g %g %g %g" % (i, para['a0'], para['k2'], para['k3'], para['k4']))
            f.write("\t\t#%8s %8s %8s\n" % (atym[i][0], atym[i][1], atym[i][2]))

        ## dihedral force field
        f.write("\n########DIHEDRALS#########\n")
        dsty = set([self.ff_db.dih_types[dih]['style'] for dih in dtym[1:]])
        dsty = list(dsty)
        if len(dsty) > 1:
            raise NotImplementedError("No support for hybrid potential.")
        if dsty[0] == 'table':
            dsty_str = self._table_potential_style_string(self.ff_db.dih_types)
        elif dsty[0] == "none":
            dsty_str = "none"
        elif dsty[0] == 'harmonic':
            dsty_str = "harmonic"
        elif dsty[0] == 'fourier':
            dsty_str = "fourier"
        else:
            raise ValueError("Invalid dihedral style.")
        f.write("#dihedral_style\t%s\n" % dsty_str)
        for i in range(1, len(dtym)):
            para = self.ff_db.dih_types[dtym[i]]
            if para['style'] == 'table':
                tmps = self._table_potential_coeff_string("dihedral", i, outputfolder, para['file'], para['keyw'])
                f.write("%s" % tmps)
            elif para['style'] == 'none':
                pass
            elif para['style'] == 'harmonic':
                f.write("dihedral_coeff\t%d %g %d %d" % (i, para['kd'], para['d'], para['n']))
            elif para['style'] == 'multi/harmonic':
                f.write("dihedral_coeff\t%d %g %g %g %g %g" % (i, *para['kd']))
            elif para['style'] == 'fourier':
                m = len(para['kd'])
                f.write("dihedral_coeff\t%d %d" % (i, m))
                for mi in range(m):
                    f.write(" %g %d %g" % (para['kd'][mi], para['n'][mi], para['d'][mi]))
            else:
                raise ValueError("Invalid dihedral style.")
            f.write("\t\t#%8s %8s %8s %8s\n" % (dtym[i][0], dtym[i][1], dtym[i][2], dtym[i][3]))
        
        f.close()


        ### read md setup file
        units = None
        temp = None
        tdamp = None
        timestep = None
        ndump = None
        dumpname = None
        relaxrun = None
        prodrun = None
        dataname = None
        with open(md_setup_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("#")[0]
                if line:
                    first, second = line.split()
                    if first == "units":
                        units = second
                    elif first == "temp":
                        temp = float(second)
                    elif first == "tdamp":
                        tdamp = float(second)
                    elif first == "timestep":
                        timestep = float(second)
                    elif first == "ndump":
                        ndump = int(second)
                    elif first == "dumpname":
                        dumpname = second
                    elif first == "relaxrun":
                        relaxrun = int(second)
                    elif first == "prodrun":
                        prodrun = int(second)
                    elif first == "dataname":
                        dataname = second
                    else:
                        print("Wrong setup file for md simulation.")
                        raise ValueError

        f = open(outputfolder + "/" + lammpsscript, "w")
        f.write("# input script generated by cgff\n")
        f.write("%-16s%s\n" % ("log", "log.run"))
        f.write("\n")
        f.write("%-16s%s\n" % ("variable", "%-16s index %s" % ("datafile", datafile)))
        f.write("%-16s%s\n" % ("variable", "%-16s index %s" % ("settingfile", settingfile)))
        ## atom type belong to polymer
        ttym_rev = {name:i for i,name in enumerate(ttym)}
        poly_atom_type = set()
        for mol in cell.mol:
            if mol.total_mass() > 1000:     ## define polymer as a chain with total mass > 1000 Da
                atom = mol.all_atom_by_id()
                for atom_ in atom.values():
                    poly_atom_type.add(ttym_rev[atom_.type])
        poly_atom_type = sorted(poly_atom_type)
        if len(poly_atom_type) > 0:
            poly_atom_type_str = ""
            for atom_type in poly_atom_type:
                poly_atom_type_str += " %d" % atom_type
            poly_atom_type_str = poly_atom_type_str[1:]
            f.write("%-16s%s\n" % ("variable", "%-16s string \"%s\"" % ("poly_type", poly_atom_type_str)))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#BASIC SETUP\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("units", units))
        f.write("%-16s%s\n" % ("atom_style", "full"))
        f.write("%-16s%s\n" % ("neighbor", "1.0 bin"))
        f.write("%-16s%s\n" % ("neigh_modify", "every 5 delay 0 check yes"))
        f.write("%-16s%s\n" % ("boundary", "p p p"))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#FORCE FIELD STYLE SETUP\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("pair_style", psty_str))
        f.write("%-16s%s\n" % ("pair_modify", "shift yes"))
        f.write("%-16s%s\n" % ("bond_style", bsty_str))
        f.write("%-16s%s\n" % ("angle_style", asty_str))
        f.write("%-16s%s\n" % ("dihedral_style", dsty_str))
        f.write("%-16s%s\n" % ("special_bonds", "lj/coul 0.0 0.0 0.0"))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#READ CONFIGURATION AND FORCE FIELD\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("read_data", "${datafile} nocoeff"))
        f.write("%-16s%s\n" % ("include", "${settingfile}"))
        f.write("\n")
        if len(poly_atom_type) > 0:
            f.write("#==============================================#\n")
            f.write("#COMPUTE AVERAGE RADIUS OF GYRATION OF POLYMERS\n")
            f.write("#==============================================#\n")
            f.write("%-16s%s\n" % ("group", "polymer type ${poly_type}"))
            f.write("%-16s%s\n" % ("compute", "molecule_chunk polymer chunk/atom molecule"))
            f.write("%-16s%s\n" % ("compute", "MolRg polymer gyration/chunk molecule_chunk"))
            f.write("%-16s%s\n" % ("variable", "MolRg_ave equal ave(c_MolRg)"))
            f.write("\n")
        f.write("#=========================================#\n")
        f.write("#OUTPUT SETUP\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("thermo", "1000"))
        if len(poly_atom_type) > 0:
            f.write("%-16s%s\n" % ("thermo_style", "custom step temp epair emol etotal press v_MolRg_ave"))
        else:
            f.write("%-16s%s\n" % ("thermo_style", "one"))
        f.write("%-16s%s\n" % ("thermo_modify", "flush yes"))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#INTEGRATION AND SIMLUATION ENSEMBLE SETUP\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("fix", "1 all nve/limit 0.05"))
        f.write("%-16s%s\n" % ("fix", "2 all langevin %g %g %g 12345" % (temp, temp, tdamp)))
        f.write("%-16s%s\n" % ("timestep", "%g" % timestep))
        f.write("#=========================================#\n")
        f.write("#RELAXATION RUN\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("velocity", "all create %g 54321 mom yes rot yes" % temp))
        f.write("%-16s%s\n" % ("run", "%d" % relaxrun))
        f.write("%-16s%s\n" % ("write_data", "relaxed_cg.dat"))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#DUMP CONFIGURATIONS\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("dump", "1 all custom %d %s id type mol x y z ix iy iz vx vy vz" % (int(prodrun / ndump), dumpname)))
        f.write("\n")
        f.write("#=========================================#\n")
        f.write("#PRODUCTION RUN\n")
        f.write("#=========================================#\n")
        f.write("%-16s%s\n" % ("reset_timestep", "0"))
        f.write("%-16s%s\n" % ("unfix", "1"))
        f.write("%-16s%s\n" % ("fix", "1 all nve"))
        #f.write("%-16s%s\n" % ("unfix", "2"))
        #f.write("%-16s%s\n" % ("fix", "1 all nvt temp %g %g %g" % (self.temp, self.temp, self.tdamp)))
        f.write("\n")
        f.write("%-16s%s\n" % ("run", "%d" % prodrun))
        f.write("\n")
        f.write("%-16s%s\n" % ("write_data", dataname))
        f.close()
