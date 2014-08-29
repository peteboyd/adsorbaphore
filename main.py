#!/usr/bin/env python
from pharma import Pharmacophore, MOFDiscovery, BindingSiteDiscovery
from faps import Structure
from config_fap import Options

from mpi_pharma import MPIPharmacophore, MPIMOFDiscovery

def add_to_pharmacophore(mof, pharma, path, energy_min, energy_max):
    faps_graph = pharma.get_main_graph(mof)
    binding_sites = BindingSiteDiscovery(path)
    site_count = 0
    if binding_sites.absl_calc_exists():
        binding_sites.co2_xyz_parser()
        binding_sites.energy_parser()
        for site in binding_sites.sites:
            try:
                el = site.pop('electrostatic')
                vdw = site.pop('vanderwaals')
                if (el + vdw) >= energy_min and (el + vdw) <= energy_max:
                    coords = np.array([site[i] for i in ['C', 'O1', 'O2']])
                    active_site = pharma.get_active_site(coords, faps_graph)
                    # set all elements to C so that the graph matching is non-discriminatory
                    #active_site.set_elem_to_C()
                    pharma.store_active_site(active_site, 
                                             name="%s.%i"%(mof.name, site_count),
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
            add_to_pharmacophore(faps_mof, pharma, path, energy_min=options.en_min, energy_max=options.en_max)
            if mofcount % 100 == 0:
                pharma.sql_active_sites.flush()
    pharma.sql_active_sites.flush()
    t2 = time()
    #if rank==0:
    #    print "number of binding sites = %i"%(pharma.site_count)
    #sys.exit()
    final_nodes, final_names = pharma.run_pharma_tree()
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
        pharma.write_final_files(final_names, final_nodes, total_site_count)
        print "Finished. Scanned %i binding sites"%(total_site_count)
        print "Reduced to %i active sites"%(len(final_nodes))
        #print "   time for initialization = %5.3f seconds"%(t2-t1)
        print "time for pharma discovery = %5.3f seconds"%(pharma.time)
        #print "                Total time = %5.3f seconds"%(t3-t1)

if __name__ == "__main__":
    main_pharma()
