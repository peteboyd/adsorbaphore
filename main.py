#!/usr/bin/env python
from pharma import Pharmacophore, MOFDiscovery, BindingSiteDiscovery
from optparse import OptionParser
from faps import Structure
from config_fap import Options
import numpy as np
from scipy.spatial import distance
from mpi_pharma import MPIPharmacophore, MPIMOFDiscovery
from time import time
import os
#MPI stuff
global MPIrank, MPIsize, comm
MPIrank, MPIsize, comm = 0, 1, None
#from mpi_pharma import MPIPharmacophore, MPIMOFDiscovery

class CommandLine(object):
    """Parse command line options and communicate directives to the program."""

    def __init__(self):
        self.commands = {}
        self.command_options()

    def command_options(self):
        usage = "%prog [options]"
        parser = OptionParser(usage=usage)
        parser.add_option("-N", "--num_mofs", action="store",
                          type="int",
                          default=8,
                          dest="nmofs",
                          help="Set the number of MOFs to scan binding sites "+
                          "for, this will be on a first-come first-serve basis. "+
                          "Default is 8 MOFs.")
        parser.add_option("--radii", action="store",
                          type="float",
                          default=4.5,
                          dest="radii",
                          help="Set the radius around each atom in the " +
                          "binding site [CO2] in which to collect framework "+
                          "atoms to be included in the 'active site'. "+
                          "Default set to 4.5 Angstroms.")
        parser.add_option("--min_atom_cutoff", action="store",
                          type="int",
                          default=9,
                          dest="mac",
                          help="Set the minimum number of atoms " +
                          "allowed when finding a maximum clique between "+
                          "two active sites. Default is 9 atoms.")
        parser.add_option("--random_seed", action="store",
                          type="int",
                          default=None,
                          dest="seed",
                          help="request the random number generation "+
                               "required to randomly pair active sites to be "+
                               "seeded with an integer. Default will seed based "+
                               "on the system time.")
        parser.add_option("--num_pass", action="store",
                          type="int",
                          dest="num_pass",
                          default=9,
                          help="Termination criteria. The program will be considered "+
                               "'done' when no new active sites are found from "+
                               "this many consecutive sets of random pairings of active sites. "+
                               "The default is 9 passes.")
        parser.add_option("--tol", action="store",
                          type="float",
                          default=0.4,
                          dest="tol",
                          help="set tolerance in angstroms for the clique "+
                               "finding algorithm. Default is 0.4 Angstroms.")
        parser.add_option("--path", action="store",
                          type="string",
                          default="/share/scratch/pboyd/binding_sites",
                          dest="search_path",
                          help="Set the search path for the binding site discovery "+
                          "process. The program performs a recursive directory search, "+
                          "so be sure to set this value to the highest possible base directory "+
                          "where the binding sites can be located. Default is set to "+
                          "'/share/scratch/pboyd/binding_sites/'")
        parser.add_option("--en_max", action="store",
                          type="float",
                          default=0.0,
                          dest="en_max",
                          help="Set the maximum energy cutoff of the binding sites to select and "+
                          "compare with. The default value is 0. "+
                          "Only binding sites with energies below this value will be considered.")
        parser.add_option("--en_min", action="store",
                          type="float",
                          default=-np.inf,
                          dest="en_min",
                          help="Set the minimum energy cutoff of the binding sites to select and "+
                          "compare with. The default value is -inf. "+
                          "Only binding sites with energies above this value will be considered.")

        parser.add_option("--mpi", action="store_true",
                          default=False,
                          dest="MPI",
                          help="Toggle the parallel version of the code. Default serial. "+
                          "***DEPRECIATED*** - found to be too slow in the current implementation.")
        (local_options, local_args) = parser.parse_args()
        self.options = local_options

def convert_to_active_site(mof, indices, original_indices, coords):
    """(ind, x, y, z, element, mof_atom_id, charge)"""
    actsit = []
    for j, id in enumerate(indices):
        mof_id = original_indices[id]
        x, y, z = coords[j][:]
        atom = mof.atoms[mof_id]
        actsit.append((j, x, y, z, atom.type, int(mof_id), atom.charge))
    return actsit, distance.cdist(coords, coords)

def add_to_pharmacophore(mof, pharma, path, options, energy_min, energy_max):
    #faps_graph = pharma.get_main_graph(mof)
    binding_sites = BindingSiteDiscovery(path)
    site_count = 0
    # compute supercell needed to support the radius in the unit cell
    supercell = mof.cell.minimum_supercell(options.radii*2)
    supercell = tuple([i if i >= 3 else 3 for i in supercell])

    original_indices, mof_coordinates = pharma.compute_supercell(mof, supercell)
    if binding_sites.absl_calc_exists():
        binding_sites.co2_xyz_parser()
        binding_sites.energy_parser()
        for site in binding_sites.sites:
            try:
                el = site.pop('electrostatic')
                vdw = site.pop('vanderwaals')
                if (el + vdw) >= energy_min and (el + vdw) <= energy_max:
                    coords = np.array([site[i] for i in ['C', 'O1', 'O2']])
                    indices, acoords = pharma.get_active_site(coords, mof_coordinates)
                    # shift by C in CO2
                    acoords -= coords[0]
                    active_site, dmatrix = convert_to_active_site(mof, indices, original_indices, acoords)
                    pharma.store_active_site(active_site, dmatrix, 
                                             name="%s.%i"%(mof.name, site_count),
                                             mof_path=os.path.join(path, mof.name+'.cif'),
                                             el_energy=el,
                                             vdw_energy=vdw)
                    # shift the co2 pos so the carbon is at the origin
                    pharma.store_co2_pos(coords-coords[0],
                                         name="%s.%i"%(mof.name, site_count))
            except KeyError:
                print "Error, could not find the binding site energies for " + \
                      "MOF %s"%(mof.name)
            site_count += 1


def main_pharma():
    options = CommandLine().options
    t1 = time()
    if options.MPI:
        pharma = MPIPharmacophore(
                           radii=options.radii,
                           min_atom_cutoff=options.mac,
                           max_pass_count=options.num_pass,
                           random_seed=options.seed,
                           tol=options.tol)
        directory = MPIMOFDiscovery(options.search_path)
    else:
        pharma = Pharmacophore(
                           radii=options.radii,
                           min_atom_cutoff=options.mac,
                           max_pass_count=options.num_pass,
                           random_seed=options.seed,
                           tol=options.tol)
        directory = MOFDiscovery(options.search_path)
    directory.dir_scan(max_mofs=options.nmofs)
    # split if mpi.
    mofcount = 0
    for mof, path in directory.mof_dirs:
        if mof:
            mofcount += 1
            faps_mof = Structure(name=mof)
            faps_mof.from_cif(os.path.join(path, mof+".cif"))
            add_to_pharmacophore(faps_mof, pharma, path, options, energy_min=options.en_min, energy_max=options.en_max)
            if mofcount % 100 == 0:
                pharma.sql_active_sites.flush()
    pharma.sql_active_sites.flush()
    t2 = time()
    #if rank==0:
    #    print "number of binding sites = %i"%(pharma.site_count)
    adsorbophores = pharma.run_pharma_tree()
    t3 = time()
    if options.MPI:
        total_site_count = comm.gather(pharma.site_count, root=0)
    else:
        # silly hack so the sum function works in the final file 
        # writing section
        total_site_count = [pharma.site_count]
    if MPIrank == 0:
        # write the pickle stuff
        total_site_count = sum(total_site_count)
        pharma.store_adsorbophores(adsorbophores)
        #pharma.write_final_files(*adsorbophores.items(), total_site_count)
        print "Finished. Scanned %i binding sites"%(total_site_count)
        print "Reduced to %i active sites"%(len(adsorbophores.keys()))
        #print "   time for initialization = %5.3f seconds"%(t2-t1)
        print "time for pharma discovery = %5.3f seconds"%(pharma.time)
        #print "                Total time = %5.3f seconds"%(t3-t1)

if __name__ == "__main__":
    main_pharma()
