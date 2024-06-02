import cgff.IBI as IBI
from mpi4py import MPI

comm = MPI.COMM_WORLD
me = comm.Get_rank()
ibi = IBI.main()

if me == 0:
    ibi.intra_mol = False
    ibi.w_pair = 99
    ibi.w_bond = 65
    ibi.w_angle = 15
    ibi.w_dihedral = 65
    ibi.smooth_mode_pair = "spline"
    ibi.smooth_mode_bond = "spline"
    ibi.smooth_mode_angle = "spline"
    ibi.smooth_mode_dihedral = "spline"

    ibi.minPratio_angle = 1e-4

    ibi.normed_rmsd = True

target_path="../coarse_grained/"

ibi.setup(run_path="./output/", target_data=target_path+"/output/cg.data", target_dump=target_path+"/output/cg.dump", cgtypemap=target_path+"/output/lmp2cgtype.map", \
    md_setup_file="input/md_setup.txt", \
    pair_setup_file="input/pair_setup.txt", bond_setup_file="input/bond_setup.txt", angle_setup_file="input/angle_setup.txt", dihedral_setup_file="input/dihedral_setup.txt", \
    pair_style="table", bond_style="table", angle_style="table", dihedral_style="multi/harmonic", \
    target_pressure=1.0, pressure_correction_scale=0.09)

ibi.run(max_iterate=30)
