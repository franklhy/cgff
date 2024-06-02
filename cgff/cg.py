import os
import copy
import numpy as np
from plato import data

def find_cg_bonds(detail_bonds, cgmap):
    """
    detail_bonds: a python list of 2-elements lists
        Bonds in the all atom configuration. Each element is (atoma_id, atomb_id)

    cgmap: a python dictionary
        key: id of atom in the detail simulation
        value: id of the corrosponding cg bead
    """
    bonds = set()    ### elements are tuples, each tuple element is a pair of id (a,b) of cg bead connected together, with a < b
    for detail_bond in detail_bonds:
        id_a = cgmap[detail_bond[0]]
        id_b = cgmap[detail_bond[1]]
        ### add atoms belonging to different cg beads are connected, then add a cg bond to this pair of cg beads
        ### atoms with cg bead id equals are ignored
        if id_a != 0 and id_b != 0:
            if id_a < id_b:
                bonds.add((id_a, id_b))
            elif id_a > id_b:
                bonds.add((id_b, id_a))
    bonds = list(bonds)
    bonds.sort()
    
    ### find bonded neighbors of each cg beads, will be used to when finding angles and dihedrals
    maxcgid = 1
    for v in cgmap.values():
        if v > maxcgid:
            maxcgid = v
    connect_dict = {i:[] for i in range(1,1 + maxcgid)}    ### key: id of cg bead, value: a list of cg beads bonded to key cg bead
    for bond in bonds:
        id_a = bond[0]
        id_b = bond[1]
        connect_dict[id_a].append(id_b)
        connect_dict[id_b].append(id_a)
    return bonds, connect_dict

def decide_cg_bond_type(bonds, cgtype):
    """
    bonds: a python list of tuple
        Each tuple element is a pair of id (a,b) of cg bead connected together, with a < b.
        It can be the first return object of the function _find_bonds
    
    cgtype: a python dictionary
        key: cg bead id
        value: cg bead type
    """
    bondtype_dict = {}    ### key: a tuple which is a pair of type (a,b) of cg bead connected together, with a <= b; value: cg bond type
    bondtypes = []
    bondtype_count = 0
    for bond in bonds:
        id_a = bond[0]
        id_b = bond[1]
        type_a = cgtype[id_a]
        type_b = cgtype[id_b]
        if type_a < type_b:
            typepair = (type_a, type_b)
        else:
            typepair = (type_b, type_a)
        if typepair not in bondtype_dict.keys():
            bondtype_count += 1
            bondtype_dict[typepair] = bondtype_count
        bondtypes.append(bondtype_dict[typepair]) 
    nbonds = len(bonds)
    cgtopo_bonds = np.column_stack((np.arange(1, nbonds + 1), bondtypes, bonds))
    return  cgtopo_bonds, bondtype_dict

def find_cg_angles(bonds, connect_dict):
    """
    bonds: a python list of tuple
        Each tuple element is a pair of id (a,b) of cg bead connected together, with a < b.
        It can be the first return object of the function _find_bonds
    
    connect_dict: a python dictionary
        key: id of cg bead
        value: a list of cg beads bonded to key cg bead
        It can be the second return object of the function _find_bonds
    """
    angles = set()    ### elements are tuples, each tuple element is a triad of id (a,b,c) of cg bead forming a cg angle, with a < c
    for bond in bonds:
        id_a = bond[0]
        id_b = bond[1]
        ### find cg beads connected to cg bead a, and build a cg angle with cg bead a at the center
        for id_c in connect_dict[id_a]:
            if id_c != id_b:
                if id_b < id_c:
                    angles.add((id_b, id_a, id_c))
                else:
                    angles.add((id_c, id_a, id_b))
        ### find cg beads connected to cg bead b, and build a cg angle with cg bead b at the center
        for id_c in connect_dict[id_b]:
            if id_c != id_a:
                if id_a < id_c:
                    angles.add((id_a, id_b, id_c))
                else:
                    angles.add((id_c, id_b, id_a))
    angles = list(angles)
    angles.sort()
    return angles

def decide_cg_angle_type(angles, cgtype):
    """
    angles: a python list of tuple
        Elements are tuples, each tuple element is a triad of id (a,b,c) of cg bead forming a cg angle, with a < c.
        It can be the return object of the function _find_angles

    cgtype: a python dictionary
        key: cg bead id
        value: cg bead type
    """
    angletype_dict = {}    ### key: a tuple which is a triad of type (a,b,c) of cg bead forming the angle, with a <= c; value: cg angle type
    angletypes = []
    angletype_count = 0
    for angle in angles:
        id_a = angle[0]
        id_b = angle[1]
        id_c = angle[2]
        type_a = cgtype[id_a]
        type_b = cgtype[id_b]
        type_c = cgtype[id_c]
        if type_a < type_c:
            typetriad = (type_a, type_b, type_c)
        else:
            typetriad = (type_c, type_b, type_a)
        if typetriad not in angletype_dict.keys():
            angletype_count += 1
            angletype_dict[typetriad] = angletype_count
        angletypes.append(angletype_dict[typetriad])
    nangles = len(angles)
    cgtopo_angles = np.column_stack((np.arange(1, nangles + 1), angletypes, angles))
    return cgtopo_angles, angletype_dict

def find_cg_dihedrals(angles, connect_dict):
    """
    angles: a python list of tuple
        Elements are tuples, each tuple element is a triad of id (a,b,c) of cg bead forming a cg angle, with a < c.
        It can be the return object of the function _find_angles
    
    connect_dict: a python dictionary
        key: id of cg bead
        value: a list of cg beads bonded to key cg bead
        It can be the second return object of the function _find_bonds
    """
    dihedrals = set()    ### elements are tuples, each tuple element is a quad of id (a,b,c,d) of cg bead forming a cg dihedral, with a < d
    for angle in angles:
        id_a = angle[0]
        id_b = angle[1]
        id_c = angle[2]
        ### find cg beads d connected to cg bead a, and build a cg dihedral (d->a->b->c if d<c, else c->b->a->d)
        for id_d in connect_dict[id_a]:
            if id_d != id_b and id_d != id_c:
                if id_d < id_c:
                    dihedrals.add((id_d, id_a, id_b, id_c))
                else:
                    dihedrals.add((id_c, id_b, id_a, id_d))
        ### find cg beads d connected to cg bead c, and build a cg dihedral (a->b->c->d if a<d, else d->c->b->a)
        for id_d in connect_dict[id_c]:
            if id_d != id_b and id_d != id_a:
                if id_a < id_d:
                    dihedrals.add((id_a, id_b, id_c, id_d))
                else:
                    dihedrals.add((id_d, id_c, id_b, id_a))
    dihedrals = list(dihedrals)
    dihedrals.sort()
    return dihedrals

def decide_cg_dihedral_type(dihedrals, cgtype):
    """
    dihedrals: a python list of tuple
        Each tuple element is a quad of id (a,b,c,d) of cg bead forming a cg dihedral, with a < d
        It can be the return object of the function _find_dihedrals
    
    cgtype: a python dictionary
        key: cg bead id
        value: cg bead type
    """
    dihedraltype_dict = {}    ### key: a tuple which is a quad of type (a,b,c,d) of cg bead forming the dihedral, with a <= d; value: cg dihedral type
    dihedraltypes = []
    dihedraltype_count = 0
    for dihedral in dihedrals:
        id_a = dihedral[0]
        id_b = dihedral[1]
        id_c = dihedral[2]
        id_d = dihedral[3]
        type_a = cgtype[id_a]
        type_b = cgtype[id_b]
        type_c = cgtype[id_c]
        type_d = cgtype[id_d]
        if type_a < type_d:
            typequad = (type_a, type_b, type_c, type_d)
        elif type_a > type_d:
            typequad = (type_d, type_c, type_b, type_a)
        elif type_b < type_c:    ### when type_a == type_b
            typequad = (type_a, type_b, type_c, type_d)
        else:
            typequad = (type_d, type_c, type_b, type_a)
        if typequad not in dihedraltype_dict.keys():
            dihedraltype_count += 1
            dihedraltype_dict[typequad] = dihedraltype_count
        dihedraltypes.append(dihedraltype_dict[typequad])
    ndihedrals = len(dihedrals)
    cgtopo_dihedrals = np.column_stack((np.arange(1, ndihedrals + 1), dihedraltypes, dihedrals))
    return cgtopo_dihedrals, dihedraltype_dict

class molecule:
    def __init__(self, atomid, atomtype, bondtype, bondatom1, bondatom2, name):
        self.atomid = copy.deepcopy(atomid)
        self.atomtype = copy.deepcopy(atomtype)
        self.bondtype = copy.deepcopy(bondtype)
        self.bondatom = np.column_stack((bondatom1, bondatom2))
        self.name = name

        ### substract an offset from all atim ids, so they start from 1
        self.id_offset = np.min(self.atomid) - 1
        self.atomid -= self.id_offset
        self.bondatom -= int(self.id_offset)

        ### check if atom ids are continuous integers
        if not np.array_equal(np.sort(self.atomid), np.arange(1, 1 + len(self.atomid))):
            print("Atom id is not continuous in one molecule.")
            print(self.atomid)
            raise RuntimeError

        ### sort atomid and atomtype
        sortid = np.argsort(self.atomid)
        self.atomid = self.atomid[sortid]
        self.atomtype = self.atomtype[sortid]
        ### sort atom id in each bond
        self.bondatom = np.sort(self.bondatom, axis=1)

        self.bond = np.column_stack((self.bondtype, self.bondatom))

    def identical_molecule_topo(self, molecule):
        '''
        Check whether two molecules are identical by atom id, atom type and bond
        '''
        if not np.array_equal(self.atomid, molecule.atomid):
            return False
        if not np.array_equal(self.atomtype, molecule.atomtype):
            return False
        if len(self.bond) != len(molecule.bond):
            return False
        this_bond = np.copy(self.bond)
        that_bond = np.copy(molecule.bond)
        while len(this_bond) != 0 and len(that_bond) != 0:
            for i in range(len(that_bond)):
                if np.array_equal(this_bond[0], that_bond[i]):
                    this_bond = np.delete(this_bond, 0, axis=0)
                    that_bond = np.delete(that_bond, i, axis=0)
                    break
                if i == len(that_bond) - 1:
                    return False
        if len(this_bond) != 0 or len(that_bond) != 0:
            print("cannot match all bonds")
            return False
        else:
            return True

    def create_cg(self, cgmap, cgtype, cgid_setoff):
        this_cgmap = np.copy(cgmap)
        sortid = np.argsort(this_cgmap[:,0])
        this_cgmap = this_cgmap[sortid]
        if not np.array_equal(this_cgmap[:,0], self.atomid):
            print("cgmap of template molecule is illegal.")
            raise RuntimeError
        this_cgmap[:,0] += self.id_offset
        mask = this_cgmap[:,1] != 0         ### ignorer atoms with cg id == 0
        this_cgmap[mask,1] += cgid_setoff

        if not np.array_equal(np.unique(cgmap[mask][:,1]), np.unique(cgtype[:,0])):
            print("cgtype of template molecule is illegal.")
            raise RuntimeError
        this_cgtype = np.copy(cgtype)
        this_cgtype[:,0] += cgid_setoff

        return this_cgmap, this_cgtype


class atomistic2cg:
    def __init__(self):
        pass

    def read_typemap(self, typemapfile, keylen):
        '''
        '''
        name_to_id = {}
        with open(typemapfile, "r") as f:
            for line in f:
                line = line.strip().split("#")[0]
                if line:
                    words = line.split()
                    if len(words) == 1 + keylen:
                        tid = int(words[0])
                        if keylen == 1:
                            name = words[1]
                        else:
                            name = tuple(words[1:])
                        name_to_id[name] = tid
                    else:
                        raise RuntimeError("Wrong key length.")
        return name_to_id

    def write_typemap(self, filename, typemap):
        '''
        typemap: list
            typemap[id] = name, where id is the lammps type id, name is the type name (a string or a tuple of strings)
            typemap[0] = None is a place holder.
        '''
        f = open(filename, "w")
        f.write("# lammps type mapping\n")
        for i in range(1, len(typemap)):
            f.write("%d\t" % i)
            if isinstance(typemap[i], str):
                f.write("%-8s\n" % typemap[i])
            else:
                for j in range(len(typemap[i])):
                    f.write("%-8s " % typemap[i][j])
                f.write("\n")
        f.close()

    def create_cg_from_template(self, detailed_data, template_molecules, template_cgmap, template_cgtype):
        ### create empty array for storage
        cgmap = []
        cgtype = []

        ### read information from detailed lammps data file
        _, _, _, _, atoms = detailed_data.snapshot()
        #bonds, angles, dihedrals, impropers = detailed_data.topology()
        bonds, _, _, _ = detailed_data.topology()
        ID, TYPE, MOL = detailed_data.map("id", "type", "mol")

        print("*****************************")
        print("Matching molecules in detailed simulation to template molecules ...")
        nmols = np.max(atoms[:,MOL]).astype(int)
        nmols_found = {template.name: 0 for template in template_molecules}
        for molid in range(1, nmols+1):
            #print("%d/%d" % (molid, nmols))
            mask = atoms[:,MOL] == molid
            if np.sum(mask) == 0:
                continue
            mol_aid = atoms[mask][:,ID]
            mol_atype = atoms[mask][:,TYPE]
            mol_bond = bonds[np.in1d(bonds[:,2], mol_aid) * np.in1d(bonds[:,3], mol_aid)]
            ##print(mol_aid, mol_atype, mol_bond)
            this_molecule = molecule(mol_aid, mol_atype, mol_bond[:,1], mol_bond[:,2], mol_bond[:,3], "Mol %d" % molid)
            ### match molecules in detailed lammps data file to template molecules
            matched = False
            for i, template in enumerate(template_molecules):
                if this_molecule.identical_molecule_topo(template):
                    ### create cg information if matched
                    this_cgmap, this_cgtype = this_molecule.create_cg(template_cgmap[i], template_cgtype[i], len(cgtype)/2)
                    cgmap.extend(this_cgmap.flatten().tolist())
                    cgtype.extend(this_cgtype.flatten().tolist())
                    nmols_found[template.name] += 1
                    matched = True
                    break
            if not matched:
                print("Cannot find the matching template for molecule no. %d" % molid)
                raise RuntimeError
        for key in nmols_found.keys():
            print("Found %d molecules in detailed simulation match the template %s" % (nmols_found[key], key))
        print("*****************************")

        cgmap = np.array(cgmap).reshape((-1,2))
        cgtype = np.array(cgtype).reshape((-1,2))
        ### sort cgmap and cgtype by first column
        cgmap = cgmap[np.argsort(cgmap[:,0])]
        cgtype = cgtype[np.argsort(cgtype[:,0])]
        return cgmap, cgtype


    def gen_cg_lammps(self, detailed_dump, detailed_data, cgmap, cgtype, typemap, output_path, ignore_charge):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        ncg = len(cgtype)

        ### read necessary information from data file of detailed simulation
        detailed_data.unwrap()
        ID, TYPE, MOL, X, Y, Z, Q = detailed_data.map("id", "type", "mol", "xu", "yu", "zu", "q")
        timestep, natoms, box, props, atoms = detailed_data.snapshot()
        detailed_bonds, _, _, _ = detailed_data.topology()
        masses = detailed_data.masses()
        masses = np.array(masses)

        ### build cg topology
        cgmap_dict = {cgmap_[0]: cgmap_[1] for cgmap_ in cgmap.astype(int)}
        cgtype_dict = {cgtype_[0]: cgtype_[1] for cgtype_ in cgtype.astype(int)}
        cgbonds, cgconnect_dict = find_cg_bonds(detailed_bonds[:,(2,3)].tolist(), cgmap_dict)
        cgtopo_bonds, cgbondtype_dict = decide_cg_bond_type(cgbonds, cgtype_dict)
        cgangles = find_cg_angles(cgbonds, cgconnect_dict)
        cgtopo_angles, cgangletype_dict = decide_cg_angle_type(cgangles, cgtype_dict)
        cgdihedrals = find_cg_dihedrals(cgangles, cgconnect_dict)
        cgtopo_dihedrals, cgdihedraltype_dict = decide_cg_dihedral_type(cgdihedrals, cgtype_dict)

        ### write type mapping for cg beads and cg topology
        alltid = sorted(typemap.values())
        if alltid == [i for i in range(1, 1 + len(typemap))]:
            atomtypemap = [None] * (1 + len(typemap))
            for k,v in typemap.items():
                atomtypemap[v] = k
        self.write_typemap(os.path.join(output_path, "lmp2cgtype.map.atom"), atomtypemap)

        bondtypemap = [()] * (1 + len(cgbondtype_dict))
        for k,v in cgbondtype_dict.items():
            bondtypemap[v] = (atomtypemap[k[0]], atomtypemap[k[1]])
        self.write_typemap(os.path.join(output_path, "lmp2cgtype.map.bond"), bondtypemap)

        angletypemap = [()] * (1 + len(cgangletype_dict))
        for k,v in cgangletype_dict.items():
            angletypemap[v] = (atomtypemap[k[0]], atomtypemap[k[1]], atomtypemap[k[2]])
        self.write_typemap(os.path.join(output_path, "lmp2cgtype.map.angle"), angletypemap)

        dihedraltypemap = [()] * (1 + len(cgdihedraltype_dict))
        for k,v in cgdihedraltype_dict.items():
            dihedraltypemap[v] = (atomtypemap[k[0]], atomtypemap[k[1]], atomtypemap[k[2]], atomtypemap[k[3]])
        self.write_typemap(os.path.join(output_path, "lmp2cgtype.map.dihedral"), dihedraltypemap)

        d = data()
        ### calculate and write information into cg data file
        atommasses = masses[atoms[:,TYPE].astype(np.int32)]
        cgatoms = np.zeros((ncg, 11))
        allpos = np.zeros((ncg, 3))
        for i in range(ncg):
            mask = cgmap[:,1] == cgtype[i,0]    # cgtype[i,0] is the id of cg bead
            atomid = cgmap[mask][:,0]
            mask = np.in1d(atoms[:,ID], atomid)
            molid = np.unique(atoms[mask][:,MOL])
            if len(molid) != 1:
                print("One coarse-grained bead should only includes atoms from the same molecule.")
                print("Fail to generate CG lammps data files.")
                raise RuntimeError
            molid = molid[0]
            q = np.sum(atoms[mask][:,Q])
            mass = np.sum(atommasses[mask])
            cgatoms[i,0] = cgtype[i,0]
            cgatoms[i,1] = cgtype[i,1]
            cgatoms[i,2] = molid
            cgatoms[i,9] = q
            cgatoms[i,10] = mass
            pos = np.average(atoms[mask][:,(X,Y,Z)], axis=0, weights=atommasses[mask])
            allpos[i] = pos
        image_flag_x = np.floor_divide(allpos[:,0] - box[0], box[3] - box[0])
        wrapped_pos_x = allpos[:,0] - image_flag_x * (box[3] - box[0])
        image_flag_y = np.floor_divide(allpos[:,1] - box[1], box[4] - box[1])
        wrapped_pos_y = allpos[:,1] - image_flag_y * (box[4] - box[1])
        image_flag_z = np.floor_divide(allpos[:,2] - box[2], box[5] - box[2])
        wrapped_pos_z = allpos[:,2] - image_flag_z * (box[5] - box[2])
        cgatoms[:,3] = wrapped_pos_x
        cgatoms[:,4] = wrapped_pos_y
        cgatoms[:,5] = wrapped_pos_z
        cgatoms[:,6] = image_flag_x
        cgatoms[:,7] = image_flag_y
        cgatoms[:,8] = image_flag_z
        props = ["id", "type", "mol", "x", "y", "z", "ix", "iy", "iz", "q"]
        d.changetimestep(timestep)
        d.changebox(box)
        d.changeatoms(cgatoms[:,:10], props)
        d.changebonds(np.array(cgtopo_bonds).astype(np.int32))
        d.changeangles(np.array(cgtopo_angles).astype(np.int32))
        d.changedihedrals(np.array(cgtopo_dihedrals).astype(np.int32))
        d.changeimpropers(np.array([]).reshape(0,0).astype(np.int32))        ### no imporper interaction in cg simulation
        ### calculate masses for each cg type
        uniq_cgtype = np.unique(cgtype[:,1])
        if not np.array_equal(uniq_cgtype, np.arange(1, len(uniq_cgtype) + 1)):
            print("Illegal type assignment of coarse-grained beads.")
            print(uniq_cgtype, np.arange(1, len(uniq_cgtype) + 1))
            raise RuntimeError
        cgmasses = np.zeros(len(uniq_cgtype) + 1)
        for t in range(1, len(uniq_cgtype) + 1):
            mask = cgatoms[:,1] == t
            mass = cgatoms[mask,10]
            uniq_mass, uniq_count = np.unique(mass, return_counts=True)
            if len(uniq_mass) != 1:
                print("Warning: There are more than one masses for cg bead of the cg type %d." % t)
                print("Masses: ", uniq_mass)
                print("Counts: ", uniq_count)
                print("Choose the most popular mass %f for cg type %d." % (uniq_mass[np.argmax(uniq_count)], t))
                cgmasses[t] = uniq_mass[np.argmax(uniq_count)]
            else:
                cgmasses[t] = uniq_mass[0]
        d.changemasses(cgmasses)
        if ignore_charge:
            d.write_data(os.path.join(output_path, "cg_withcharge.data"), "full")
            cgatoms[:,9] = 0
            d.changeatoms(cgatoms[:,:10], props)
            d.write_data(os.path.join(output_path, "cg.data"), "full")
            print("CG data file cg.data (with the charge on all cg beads set to 0) and " \
                + "cg.data_withcharge is generated.")
        else:
            d.write_data(os.path.join(output_path, "cg.data"), "full")
            print("CG data file cg.data is generated.")

        ### cg lammps dump file
        if os.path.isfile(detailed_dump):
            detailed_data.set_traj_file(detailed_dump)
            alltimes = detailed_data.traj_time()
            for step in range(len(alltimes)):
                ### read information from dump file of detailed simulation
                detailed_data.load_traj(step)
                detailed_data.unwrap()
                ID, TYPE, MOL, X, Y, Z = detailed_data.map("id", "type", "mol", "xu", "yu", "zu")
                idmapping_atom = detailed_data.idtoindex_atom()
                timestep,natoms,box,props,atoms = detailed_data.snapshot()
                atommasses = masses[atoms[:,TYPE].astype(np.int32)]
                ### calculate and write information into cg dump file
                cgatoms = np.zeros((ncg,9))
                allpos = np.zeros((ncg, 3))
                for i in range(ncg):
                    #print("%d/%d" % (i,ncg))
                    mask = cgmap[:,1] == cgtype[i,0]    # cgtype[i,0] is the id of cg bead
                    atomid = cgmap[mask][:,0]
                    atomindex = idmapping_atom[atomid.astype(int)]
                    molid = np.unique(atoms[atomindex][:,MOL])
                    if len(molid) != 1:
                        print("One coarse-grained bead should only includes atoms from the same molecule.")
                        print("Fail to generate CG lammps dump files.")
                        raise RuntimeError
                    molid = molid[0]
                    cgatoms[i,0] = cgtype[i,0]
                    cgatoms[i,1] = cgtype[i,1]
                    cgatoms[i,2] = molid
                    pos = np.average(atoms[atomindex][:,(X,Y,Z)], axis=0, weights=atommasses[atomindex])
                    allpos[i] = pos
                image_flag_x = np.floor_divide(allpos[:,0] - box[0], box[3] - box[0])
                wrapped_pos_x = allpos[:,0] - image_flag_x * (box[3] - box[0])
                image_flag_y = np.floor_divide(allpos[:,1] - box[1], box[4] - box[1])
                wrapped_pos_y = allpos[:,1] - image_flag_y * (box[4] - box[1])
                image_flag_z = np.floor_divide(allpos[:,2] - box[2], box[5] - box[2])
                wrapped_pos_z = allpos[:,2] - image_flag_z * (box[5] - box[2])
                cgatoms[:,3] = wrapped_pos_x
                cgatoms[:,4] = wrapped_pos_y
                cgatoms[:,5] = wrapped_pos_z
                cgatoms[:,6] = image_flag_x
                cgatoms[:,7] = image_flag_y
                cgatoms[:,8] = image_flag_z
                props = ["id", "type", "mol", "x", "y", "z", "ix", "iy", "iz",]
                d.changetimestep(timestep)
                d.changebox(box)
                d.changeatoms(cgatoms, props)
                if step == 0:
                    write_mode = "w"
                else:
                    write_mode = "a"
                d.write_dump(os.path.join(output_path, "cg.dump"), props, write_mode)
        else:
            print("Cannot find detailed dump file %s." % detailed_dump)

    def main(self, whichmode, detailed_datafile, detailed_dumpfile, template_datafiles, cgmap_files, cgtype_files, typemap_files, output_path, \
        ignore_charge=True):
        print("Coarse graining all-atom simulation....")
        detailed_data = data()
        detailed_data.read(detailed_datafile)
        if whichmode == "mol":
            ### t_ for template molecules and their cg information
            t_data = data()
            t_molecule = []
            t_cgmap = []
            t_cgtype = []
            typemap = {}
            for i,t_file in enumerate(template_datafiles):
                ### template molecule
                t_data.read(t_file)
                ID, TYPE = t_data.map("id", "type")
                _, _, _, _, t_atoms = t_data.snapshot()
                #t_bond, t_angle, t_dihedral, t_improper = t_data.topology()
                t_bond, _, _, _ = t_data.topology()
                this_molecule = molecule(t_atoms[:,ID], t_atoms[:,TYPE], t_bond[:,1], t_bond[:,2], t_bond[:,3], t_file)
                t_molecule.append(this_molecule)

                ### cg information for the template molecule
                t_cgmap_i = np.loadtxt(cgmap_files[i], ndmin=2)
                t_cgmap.append(t_cgmap_i)
                t_cgtype_i = np.loadtxt(cgtype_files[i], ndmin=2)
                t_cgtype.append(t_cgtype_i)
                t_typemap_i = self.read_typemap(typemap_files[i], keylen=1)
                for key in t_typemap_i.keys():
                    if key in typemap.keys() and t_typemap_i[key] != typemap[key]:
                        raise RuntimeError("Same cg type is assigned to different lammps type id.")
                    else:
                        typemap[key] = t_typemap_i[key]

            cgmap, cgtype = self.create_cg_from_template(detailed_data, t_molecule, t_cgmap, t_cgtype)

        elif whichmode == "all":
            if isinstance(cgmap_files, str) and isinstance(cgtype_files, str) and isinstance(typemap_files, str):
                cgmap = np.loadtxt(cgmap_files, ndmin=2)
                cgtype = np.loadtxt(cgtype_files, ndmin=2)
                typemap = self.read_typemap(typemap_files, keylen=1)
            else:
                raise RuntimeError

        self.gen_cg_lammps(detailed_dumpfile, detailed_data, cgmap, cgtype, typemap, output_path, ignore_charge)
