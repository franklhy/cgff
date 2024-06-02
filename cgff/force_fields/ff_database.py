class ff_database:
    def __init__(self):
        self.atm_types = {}
        self.par_types = {}
        self.bon_types = {}
        self.ang_types = {}
        self.dih_types = {}
        self.imp_types = {}

    def add_dummy_type(self, fftype, name):
        '''
        Add dummy force field. This is used when certain force field type is missing.
        '''
        if fftype == "atom":
            if not isinstance(name, str):
                raise ValueError("Input name for atom type should be a string.")
            elif name in self.atm_types.keys():
                raise ValueError("Force field for atom type %s exist." % repr(name))
            else:
                self.atm_types[name] = {'m': 0.0, 'eps': 0.0, 'r': 0.0}
        elif fftype == "pair":
            if not isinstance(name, tuple) or len(name) != 2:
                raise ValueError("Input name for pair type should be a tuple of 2 strings.")
            elif name in self.par_types.keys():
                raise ValueError("Force field for pair type %s exist." % repr(name))
            else:
                self.par_types[name] = {'style': "lj/cut", 'eps': 0.0, 'sig': 0.0, 'rcut': 0.0}
        elif fftype == "bond":
            if not isinstance(name, tuple) or len(name) != 2:
                raise ValueError("Input name for bond type should be a tuple of 2 strings.")
            elif name in self.bon_types.keys():
                raise ValueError("Force field for bond type %s exist." % repr(name))
            else:
                self.bon_types[name] = {'style': "harmonic", 'kb': 0.0, 'b0': 0.0}
        elif fftype == "angle":
            if not isinstance(name, tuple) or len(name) != 3:
                raise ValueError("Input name for angle type should be a tuple of 3 strings.")
            elif name in self.ang_types.keys():
                raise ValueError("Force field for angle type %s exist." % repr(name))
            else:
                self.ang_types[name] = {'style': "charmm", 'ka': 0.0, 'a0': 0.0}
        elif fftype == "dihedral":
            if not isinstance(name, tuple) or len(name) != 4:
                raise ValueError("Input name for dihedral type should be a tuple of 4 strings.")
            elif name in self.dih_types.keys():
                raise ValueError("Force field for dihedral type %s exist." % repr(name))
            else:
                self.dih_types[name] = {'style': "charmm", 'kd': 0.0, 'n': 0.0, 'delta': 0.0}
        elif fftype == "improper":
            if not isinstance(name, tuple) or len(name) != 4:
                raise ValueError("Input name for improper type should be a tuple of 4 strings.")
            elif name in self.imp_types.keys():
                raise ValueError("Force field for improper type %s exist." % repr(name))
            else:
                self.imp_types[name] = {'style': "harmonic", 'ki': 0.0, 'x0': 0.0}
        else:
            raise ValueError("Invalid force field type \'%s\'. " % fftype + "Please choose from " + \
                "\'atom\', \'bond\', \'angle\', \'dihedral\' and \'improper\'.")

    def write_forecefield_database(self, ff_db_name="forcefield.py"):
        if ff_db_name is not None:
            fout = open(ff_db_name, "w")
            print("# DATABASE DICTIONARIES", file=fout)
            print("atm_types = {}", file=fout)
            print("par_types = {}", file=fout)
            print("bon_types = {}", file=fout)
            print("ang_types = {}", file=fout)
            print("dih_types = {}", file=fout)
            print("imp_types = {}", file=fout)
            print("\n", file=fout)

            print("#####AtomsSTART#######", file=fout)
            print("###     Atoms      ###", file=fout)
            print("######################", file=fout)
            for key in self.atm_types.keys():
                print("%-20s" % ("atm_types[\'%s\']" % key), "=", self.atm_types[key], file=fout)
            print("######AtomsEND########", file=fout)
            print("\n", file=fout)

            print("#####PairsSTART#######", file=fout)
            print("###      Pairs     ###", file=fout)
            print("######################", file=fout)
            for key in self.par_types.keys():
                print("%-35s" % ("par_types[(\'%s\',\'%s\')]" % (*key,)), "=", self.par_types[key], file=fout)
            print("######PairsEND########", file=fout)
            print("\n", file=fout)

            print("#####BondsSTART#######", file=fout)
            print("###      Bonds     ###", file=fout)
            print("######################", file=fout)
            for key in self.bon_types.keys():
                print("%-35s" % ("bon_types[(\'%s\',\'%s\')]" % (*key,)), "=", self.bon_types[key], file=fout)
            print("######BondsEND########", file=fout)
            print("\n", file=fout)

            print("#####AnglesSTART#######", file=fout)
            print("###      Angles     ###", file=fout)
            print("#######################", file=fout)
            for key in self.ang_types.keys():
                print("%-45s" % ("ang_types[(\'%s\',\'%s\',\'%s\')]" % (*key,)), "=", self.ang_types[key], file=fout)
            print("#####AnglesEND########", file=fout)
            print("\n", file=fout)

            print("#####DihedralsSTART#######", file=fout)
            print("###     Dihedrals      ###", file=fout)
            print("##########################", file=fout)
            for key in self.dih_types.keys():
                print("%-55s" % ("dih_types[(\'%s\',\'%s\',\'%s\',\'%s\')]" % (*key,)), "=", self.dih_types[key], file=fout)
            print("######DihedralsEND########", file=fout)
            print("\n", file=fout)

            print("#####ImpropersSTART#######", file=fout)
            print("###     Impropers      ###", file=fout)
            print("##########################", file=fout)
            for key in self.imp_types.keys():
                print("%-55s" % ("imp_types[(\'%s\',\'%s\',\'%s\',\'%s\')]" % (*key,)), "=", self.imp_types[key], file=fout)
            print("######ImpropersEND########", file=fout)
            print("\n", file=fout)

            fout.close()
