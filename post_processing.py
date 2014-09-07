import numpy as np
from config_fap import Options
from StringIO import StringIO
from faps import Structure, Atom, Cell, mk_gcmc_control, Guest
import subprocess
from sql_backend import Data_Storage, SQL_Pharma, SQL_ActiveSite, SQL_ActiveSiteAtoms, SQL_Distances, SQL_ActiveSiteCO2
from sql_backend import SQL_Adsorbophore, SQL_AdsorbophoreSite, SQL_AdsorbophoreSiteIndices
from math import pi
import os
import sys
ANGS2BOHR = 1.889725989
ATOMIC_NUMBER = [
    "ZERO", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Uut", "Uuq", "Uup", "Uuh", "Uuo"]

class Fastmc_run(object):
    """Compute a single point energy from a FASTMC job"""
    def __init__(self, supercell=(1, 1, 1)):
        self._fmcexe = '/home/pboyd/codes_in_development/fastmc/gcmc.x'
        self.options = Options()
        self._set_options(supercell)
        self.struct = Structure(name='pharma')

    def _set_options(self, supercell):
        hack = '[job_config]\n'
        hack += 'mc_zero_charges = False\n'
        hack += 'mc_supercell = (%i, %i, %i)\n'%(supercell)
        hack += 'mc_cutoff = 12.5\n'
        hack += 'mc_prod_steps = 1\n'
        hack += 'mc_eq_steps = 1\n'
        hack += 'mc_numguests_freq = 1\n'
        hack += 'mc_history_freq = 1\n'
        hack += 'mc_jobcontrol = False\n'
        hack += 'mc_probability_plot = False\n'
        hack += 'fold = False\n'
        hack += 'mc_probability_plot_spacing = 0.1'
        hack = StringIO(hack)
        self.options.job_ini.readfp(hack)
        self.guest_dic = {}

    def add_guest(self, position):
        guest = Guest(ident='CO2')
        self.struct.guests = [guest]
        self.guest_dic = {guest.ident: [position]}

    def add_fragment(self, elems, coords, charges, lattice):
        cell = Cell()
        cell._cell = np.array(lattice)
        self.struct.cell = cell
        for elem, coord, charge in zip(elems, coords, charges):
            atom = Atom(at_type=elem, pos=coord, parent=self.struct, charge=charge)
            self.struct.atoms.append(atom)

    def run_fastmc(self):
        config, field = self.struct.to_config_field(self.options, fastmc=True, include_guests=self.guest_dic)
        control = mk_gcmc_control(298.0, 0.15, self.options, self.struct.guests)
        control.pop(-1)
        control.append('single point\nfinish')
        file = open("CONFIG", "w")
        file.writelines(config)
        file.close()
        file = open("CONTROL", "w")
        file.writelines(control)
        file.close()
        file = open("FIELD", "w")
        file.writelines(field)
        file.close()
        p = subprocess.Popen([self._fmcexe], stdout=subprocess.PIPE)
        q = p.communicate()

    def obtain_energy(self):
        outfile = open('OUTPUT', 'r')
        outlines = outfile.readlines()
        outfile.close()
        vdw = 0.
        elstat = 0.
        for line in outlines:
            if 'van der Waals energy :' in line:
                try:
                    vdw = float(line.split()[5])
                except IndexError:
                    print "Warning! could not get the vdw energy from one of the runs"
                    vdw = 0.
            elif 'Electrostatic energy :' in line:
                try:
                    elstat = float(line.split()[3])
                except IndexError:
                    print "Warning! could not get the electrostatic energy from one of the runs"
                    elstat = 0.
        return vdw, elstat

    def clean(self, i=0):
        #shutil.copyfile("CONFIG", "CONFIG.%i"%i)
        #shutil.copyfile("CONTROL", "CONTROL.%i"%i)
        #shutil.copyfile("FIELD", "FIELD.%i"%i)
        #shutil.copyfile("OUTPUT", "OUTPUT.%i"%i)
        os.remove("CONFIG")
        os.remove("CONTROL")
        os.remove("FIELD")
        os.remove("OUTPUT")

class PairDistFn(object):
    """Creates all radial distributions from a set of binding sites to
    their functional groups.

    """
    def __init__(self, bins=10, max=10):
        self.label_counts = {}
        self.bins = bins
        self.max_dist = max
        self.vol_av = []
        self._C_distances = {}
        self._O_distances = {}

    def bin_co2_distr(self, sites, net, energy_cut=None):
        """take the current MOF and obtain the radial distributions."""
        cells = self._determine_supercell()
        cell = net.cell
        local_label_count = {}
        self.vol_av.append(net.volume)
        for site in sites:
            if energy_cut is not None:
                if (site['electrostatic'] + site['vanderwaals']) > -abs(energy_cut):
                    continue
            for label, coord in net.nodes:
                if not label.startswith('o') and not label.startswith('m'):
                    self.label_counts.setdefault(label, {})
                    self.label_counts[label].setdefault('nconfig', 0)
                    self.label_counts[label].setdefault('nparticle', 0)
                    self.label_counts[label]['nconfig'] += 1
                    local_label_count.setdefault(label, 0)
                    local_label_count[label] += 1
                    # create periodic images
                    fcoord = np.dot(net.icell, coord)
                    images = np.array(fcoord) + \
                             list(itertools.product(cells, repeat=3))
                    # set to cartesian coordinates
                    cart_images = np.dot(images, cell)
                    # compute distances to C, O
                    coords = np.array([site['C'], site['O1'], site['O2']])
                    distmat = self._eval_distances(coords, cart_images)
                    self._C_distances.setdefault(label, [])
                    self._O_distances.setdefault(label, [])
                    for dist in distmat[0]:
                        self.label_counts[label]['nparticle'] += 1
                        self._C_distances[label].append(dist)
                    for dist1, dist2 in zip(distmat[1], distmat[2]):
                        self._O_distances[label].append(dist1)
                        self._O_distances[label].append(dist2)

        for label, val in local_label_count.items():
            # update densities with the unit cell density
            self.label_counts[label].setdefault('rho', [])
            self.label_counts[label]['rho'].append((float(val) / net.volume))

    def _eval_distances(self, coord1, coord2):
        return distance.cdist(coord1, coord2)

    def _get_histogram(self, label, key="C"):
        print "number of entries: %i"%(len(self._C_distances[label]))
        if key == "C":
            hist, bin_edges =  np.histogram(self._C_distances[label],
                                            bins=self.bins,
                                            range=(0., self.max_dist))
        elif key == "O":
            hist, bin_edges =  np.histogram(self._O_distances[label],
                                            bins=self.bins,
                                            range=(0., self.max_dist))
        try:
            self.dr
        except AttributeError:
            self.dr = np.diff(bin_edges)[0]
        return hist

    def _get_norm(self, label):
        """Need to fix rho to a specific user-defined value."""
        rho = np.mean(self.label_counts[label]['rho']) 
        nconfigs = self.label_counts[label]['nconfig']
        nparticles = self.label_counts[label]['nparticle']
        return 2. * pi * self.dr * rho * float(nparticles) / float(nconfigs) \
               * float(nconfigs)

    def _determine_supercell(self):
        """Currently set to 3x3x3. Determine the supercell that would
        fit a box of size self.max_dist"""
        return (-1, 0, 1)

    def _radii(self, i):
        return self.dr*(float(i) - 0.5)

    def compute_rdf(self, label, key="C"):
        hist = self._get_histogram(label, key=key)
        norm = self._get_norm(label)
        rdf = [hist[i]/norm/(self._radii(i)*self._radii(i) + self.dr*self.dr/12.)
                for i in range(self.bins)]
        return rdf

class PostRun(object):
   
    def __init__(self, options):
        self.options = options
        self.check_for_active_sites_file()
        self.check_for_adsorbophore_file()
        self.adsorbophore_db = Data_Storage(self.options.adsorbophore_file[:-3])
        self.active_sites_db = Data_Storage(self.options.active_sites_file[:-3])

    def check_for_active_sites_file(self):
        if self.options.active_sites_file:
            if not os.path.isfile(self.options.active_sites_file):
                print "Could not find %s, exiting."%self.options.active_sites_file
                sys.exit()
        else:
            file = [i for i in os.listdir(self.options.working_dir) 
                    if i.startswith('active_sit') and i.endswith('.db')]
            try:
                self.options.active_sites_file = file[0]
            except IndexError:
                print "Could not find an active_sites database file in %s, exiting."%(
                        self.options.working_dir)
                sys.exit()
    
    def check_for_adsorbophore_file(self):
        if self.options.adsorbophore_file:
            if not os.path.isfile(self.options.adsorbophore_file):
                print "Could not find %s, exiting."%self.options.adsorbophore_file
                sys.exit()
        else:
            file = [i for i in os.listdir(self.options.working_dir) 
                    if i.startswith('adsorbopho') and i.endswith('.db')]
            try:
                self.options.adsorbophore_file = file[0]
            except IndexError:
                print "Could not find an adsorbophore database file in %s, exiting."%(
                        self.options.working_dir)
                sys.exit()

    def adsorbophore_from_sql(self, rank):
        count = self.adsorbophore_db.ads_count()
        if rank > count:
            print "Rank %i is too big, the adsorbophore database only goes up to %i"%(rank, count)
            sys.exit()
        ads = self.adsorbophore_db.get_adsorbophore(rank)
        return ads

    def centre_of_atoms(self, coords):
        return np.average(coords, axis=0)

    def obtain_error(self, rank, debug_ind=0):
        adsorbophore = self.adsorbophore_from_sql(rank)
        # get the first adsorbophore as the representative one for the adsorbophore
        base_ads = adsorbophore.active_sites[0]
        # get the indices for the adsorbophore
        indices = [i.index for i in base_ads.indices]
        # obtain the original active site from the MOF
        base_site = self.active_sites_db.get_active_site(base_ads.name)
        # get the coordinates of the relevant atoms from the active site
        base_coords = np.array([[base_site.atoms[j].x, base_site.atoms[j].y, base_site.atoms[j].z] for j in indices])
        base_coords -= self.centre_of_atoms(base_coords)
        #base_elements = [base_site.atoms[i].elem.encode('ascii', 'ignore') for i in indices]
        mean_errors = []
        #f = open('debug.xyz', 'w')
        #f.writelines("%i\nbase\n"%(len(base_elements)))
        #for atom, coord in zip(base_elements, base_coords):
        #    f.writelines("%s %6.3f %6.3f %6.3f\n"%(atom, coord[0], coord[1], coord[2]))

        for site in adsorbophore.active_sites[1:]:
            indices = [i.index for i in site.indices]
            act_sit = self.active_sites_db.get_active_site(site.name)
            act_coords = np.array([[act_sit.atoms[j].x, act_sit.atoms[j].y, act_sit.atoms[j].z] for j in indices])
            #act_elements = [act_sit.atoms[i].elem.encode('ascii', 'ignore') for i in indices]
            act_coords -= self.centre_of_atoms(act_coords)
            R = rotation_from_vectors(act_coords[:], base_coords[:])
            act_coords = np.dot(R[:3,:3], act_coords.T).T
            #f.writelines("%i\nother\n"%(len(act_elements)))
            #for atom, coord in zip(act_elements, act_coords):
            #    f.writelines("%s %6.3f %6.3f %6.3f\n"%(atom, coord[0], coord[1], coord[2]))
            for p, q in zip(base_coords, act_coords):
                mean_errors.append((p-q)**2)
        #f.close()
        return np.sqrt(np.mean(mean_errors))

    def get_grid_indices(self, coord, (nx, ny, nz)):
        vect = coord / np.array([nx, ny, nz])[:, None]
        return np.floor(vect).astype(int)

    def increment_grid(self, grid, inds, en=None):
        if len(inds.shape) == 1:
            try:
                if en is None:
                    grid[np.split(inds, 3)] += 1.
                else:
                    grid[np.split(inds, 3)] += en
            except IndexError:
                print "Warning, could not include one of the binding sites" + \
                " in the prob distribution due to distance"

        elif len(inds.shape) == 2:
            for i in inds:
                try:
                    if en is None:
                        grid[np.split(i, 3)] += 1.
                    else:
                        grid[np.split(i, 3)] += en 

                except IndexError:
                    print "Warning, could not include one of the binding sites" + \
                    " in the prob distribution due to distance"

        else:
            print "WARNING: unrecognized indices passed to grid storage routine!"
            print inds
    
    def obtain_co2_fragment_energies(self, elems, coords, charges, cell, co2, i=0):
        cell = (cell.T * np.array([5.,5.,5.])).T
        fastmc = Fastmc_run(supercell=(1,1,1))
        fastmc.add_guest(co2)
        fastmc.add_fragment(elems, coords, charges, cell)
        fastmc.run_fastmc()
        vdw, el = fastmc.obtain_energy()
        fastmc.clean(i)
        return vdw*4.184, el*4.184

    def obtain_total_energy_distribution(self, rank):
        gride = np.zeros((ngridx, ngridy, ngridz))
        gridecount = np.ones((ngridx, ngridy, ngridz))
        _2radii = self.options.radii*2. + 2.

        shift_vector = np.array([_2radii/2., _2radii/2., _2radii/2.])
        nx, ny, nz = _2radii/float(ngridx), _2radii/float(ngridy), _2radii/float(ngridz)
        
        adsorbophore = self.adsorbophore_from_sql(rank)
        # get the first adsorbophore as the representative one for the adsorbophore
        base_ads = adsorbophore.active_sites[0]
        # get the indices for the adsorbophore
        indices = [i.index for i in base_ads.indices]
        # obtain the original active site from the MOF
        base_site = self.active_sites_db.get_active_site(base_ads.name)
        base_elem = [base_site.atoms[i].elem.encode('ascii', 'ignore') for i in indices]
        # get the coordinates of the relevant atoms from the active site
        base_coords = np.array([[base_site.atoms[j].x, base_site.atoms[j].y, base_site.atoms[j].z] for j in indices])
        base_charges = np.array([base_site.atoms[j].charge for j in indices])
        T = self.centre_of_atoms(base_coords)
        base_coords -= T

        base_eng = base_site.vdweng + base_site.eleng
        size = 0
        if base_eng <= self.options.en_max and base_eng >= self.options.en_min:
            mofname = os.path.split(base_site.mofpath)[-1][:-3]
            struct = Structure(mofname)
            struct.from_cif(base_site.mofpath)
            co2 = self.return_co2_array(base_site)
            co2 -= T
            vdw, el = self.obtain_co2_fragment_energies(base_elem, base_coords,
                    base_charges, struct.cell.cell, co2)
            inds = self.get_grid_indices(co2 + shift_vector, (nx,ny,nz))
            self.increment_grid(gride, inds[0], en=vdw+el)
            self.increment_grid(gridecount, inds[0]) 
            size += 1

        for site in adsorbophore.active_sites[1:]:
            indices = [i.index for i in site.indices]
            act_sit = self.active_sites_db.get_active_site(site.name)
            act_elem = [act_sit.atoms[i].elem.encode('ascii', 'ignore') for i in indices]
            act_coords = np.array([[act_sit.atoms[j].x, act_sit.atoms[j].y, act_sit.atoms[j].z] for j in indices])
            act_charges = np.array([act_sit.atoms[j].charge for j in indices])
            site_eng = act_sit.vdweng + act_sit.eleng
            if site_eng <= self.options.en_max and site_eng >= self.options.en_min:
                mofname = os.path.split(act_sit.mofpath)[-1][:-3]
                struct = Structure(mofname)
                struct.from_cif(act_sit.mofpath)

                T = self.centre_of_atoms(act_coords)
                act_coords -= T 
                R = rotation_from_vectors(act_coords[:], base_coords[:])

                co2 = self.return_co2_array(act_sit)
                co2 -= T
                co2 = np.dot(R[:3,:3], co2.T).T
                vdw, el = self.obtain_co2_fragment_energies(act_elem, act_coords,
                        act_charges, struct.cell.cell, co2)
                inds = self.get_grid_indices((co2+shift_vector), (nx, ny, nz))
                self.increment_grid(gride, inds[0], en=vdw+el)
                self.increment_grid(gridecount, inds[0]) 
                size += 1
        string_gride = self.get_cube_format(base_elem, base_coords, gride, size, gridecount, 
                                            ngridx, ngridy, ngridz)
        
        ecube = open('rank%i_TOTEN.cube'%(rank), 'w')
        ecube.writelines(string_gride)
        ecube.close()


    
    def obtain_split_energy_distribution(self, rank):
        gridevdw = np.zeros((ngridx, ngridy, ngridz))
        grideel = np.zeros((ngridx, ngridy, ngridz))
        gridevdwcount = np.ones((ngridx, ngridy, ngridz))
        grideelcount = np.ones((ngridx, ngridy, ngridz))
        _2radii = self.options.radii*2. + 2.

        shift_vector = np.array([_2radii/2., _2radii/2., _2radii/2.])
        nx, ny, nz = _2radii/float(ngridx), _2radii/float(ngridy), _2radii/float(ngridz)
        #evdw /= vdweng
        #eel /= eleng
        #self.increment_grid(gridevdw, inds[0], en=evdw)
        #self.increment_grid(grideel, inds[0], en=eel)

    def obtain_co2_distribution(self, rank):
        ngridx, ngridy, ngridz = [self.options.num_gridpoints]*3
        gridc = np.zeros((ngridx, ngridy, ngridz))
        grido = np.zeros((ngridx, ngridy, ngridz))

        _2radii = self.options.radii*2. + 2.
        # Because the cube file puts the pharmacophore in the middle of the box,
        # we need to shift the CO2 distributions to centre at the middle of the box
        # this was originally set to the radial cutoff distance of the initial
        # pharmacophore 
        shift_vector = np.array([_2radii/2., _2radii/2., _2radii/2.])
        nx, ny, nz = _2radii/float(ngridx), _2radii/float(ngridy), _2radii/float(ngridz)
        
        adsorbophore = self.adsorbophore_from_sql(rank)
        # get the first adsorbophore as the representative one for the adsorbophore
        base_ads = adsorbophore.active_sites[0]
        # get the indices for the adsorbophore
        indices = [i.index for i in base_ads.indices]
        # obtain the original active site from the MOF
        base_site = self.active_sites_db.get_active_site(base_ads.name)
        # get the coordinates of the relevant atoms from the active site
        base_coords = np.array([[base_site.atoms[j].x, base_site.atoms[j].y, base_site.atoms[j].z] for j in indices])
        base_elem = [base_site.atoms[i].elem.encode('ascii', 'ignore') for i in indices]
        T = self.centre_of_atoms(base_coords)
        base_coords -= T

        base_eng = base_site.vdweng + base_site.eleng
        size = 0
        if base_eng <= self.options.en_max and base_eng >= self.options.en_min:
            co2 = self.return_co2_array(base_site)
            inds = self.get_grid_indices(co2 - T + shift_vector, (nx,ny,nz))
            self.increment_grid(gridc, inds[0])
            self.increment_grid(grido, inds[1:])
            size += 1

        for site in adsorbophore.active_sites[1:]:
            indices = [i.index for i in site.indices]
            act_sit = self.active_sites_db.get_active_site(site.name)
            act_coords = np.array([[act_sit.atoms[j].x, act_sit.atoms[j].y, act_sit.atoms[j].z] for j in indices])

            site_eng = act_sit.vdweng + act_sit.eleng

            if site_eng <= self.options.en_max and site_eng >= self.options.en_min:
                T = self.centre_of_atoms(act_coords)
                act_coords -= T 
                R = rotation_from_vectors(act_coords[:], base_coords[:])

                co2 = self.return_co2_array(act_sit)
                co2 -= T
                co2 = np.dot(R[:3,:3], co2.T).T
                inds = self.get_grid_indices((co2+shift_vector), (nx, ny, nz))
                self.increment_grid(gridc, inds[0])
                self.increment_grid(grido, inds[1:])
                size += 1

        string_gridc = self.get_cube_format(base_elem, base_coords, gridc, size, float(size),
                                                ngridx, ngridy, ngridz)
        string_grido = self.get_cube_format(base_elem, base_coords, grido, size, float(size), 
                                                ngridx, ngridy, ngridz)
        ccube = open('rank%i_C.cube'%(rank), 'w')
        ccube.writelines(string_gridc)
        ccube.close()

        ocube = open('rank%i_O.cube'%(rank), 'w')
        ocube.writelines(string_grido)
        ocube.close()


           #atms = [q[ii] for q in sites]
           #co2, vdweng, eleng = self.get_active_site_from_sql(name[ii])
           #try:
           #    site = self._active_sites[name[ii]]
           #except KeyError:
           #    site = self.get_active_site_graph_from_sql(name[ii])
           #match = site % atms
           #T = match.centre_of_atoms[:3].copy()
           #match.shift_by_centre_of_atoms()
           #R = rotation_from_vectors(match._coordinates[:], 
           #                          base._coordinates[:])
           ##R = rotation_from_vectors(base._coordinates[:], 
           ##                          match._coordinates[:])
           #match.rotate(R)
           ##co2 = self._co2_sites[name[ii]].copy()
           #co2 -= T
           #co2 = np.dot(R[:3,:3], co2.T).T
           #inds = self.get_grid_indices((co2+shift_vector), (nx,ny,nz))
           #self.increment_grid(gridc, inds[0])
           #self.increment_grid(grido, inds[1:])
           #evdw, eel = self.obtain_co2_fragment_energies(name[ii], match, co2, ii)
           ##evdw /= vdweng
           ##eel /= eleng
           ##self.increment_grid(gridevdw, inds[0], en=evdw)
           ##self.increment_grid(grideel, inds[0], en=eel)
           #self.increment_grid(gride, inds[0], en=eel+evdw)
           ##self.increment_grid(gridevdwcount, inds[0])
           ##self.increment_grid(grideelcount, inds[0]) 
           #self.increment_grid(gridecount, inds[0]) 
           ## bin the x, y, and z
        #string_gridc = self.get_cube_format(base, gridc, len(name), float(len(name)),
        #                                        ngridx, ngridy, ngridz)
        #string_grido = self.get_cube_format(base, grido, len(name), float(len(name)), 
        #                                        ngridx, ngridy, ngridz)
        ##string_gridevdw = self.get_cube_format(base, gridevdw, len(name), gridevdwcount,
        ##                                        ngridx, ngridy, ngridz)
        ##string_grideel = self.get_cube_format(base, grideel, len(name), grideelcount,
        ##                                        ngridx, ngridy, ngridz)
        #string_gride = self.get_cube_format(base, gride, len(name), gridecount,
        #                                    ngridx, ngridy, ngridz)
        #return string_gridc, string_grido, string_gride #string_gridevdw, string_grideel

    def return_co2_array(self, active_site):
        c = active_site.co2[0]
        return np.array([[c.cx, c.cy, c.cz],
                         [c.o1x, c.o1y, c.o1z],
                         [c.o2x, c.o2y, c.o2z]])

    def get_cube_format(self, elem, coords, grid, count, avg, ngridx, ngridy, ngridz):
        # header
        str = "Clique containing %i binding sites\n"%count
        str += "outer loop a, middle loop b, inner loop c\n"
        str += "%6i %11.6f %11.6f %11.6f\n"%(len(elem), 0., 0., 0.)
        _2radii = (self.options.radii*2. + 2.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, _2radii*ANGS2BOHR/float(ngridx), 0., 0.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, 0., _2radii*ANGS2BOHR/float(ngridy), 0.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, 0., 0., _2radii*ANGS2BOHR/float(ngridz))
        vect = np.array([_2radii/2., _2radii/2., _2radii/2.]) 
        for atom, coord in zip(elem, coords):
            atm = ATOMIC_NUMBER.index(atom)
            shifted_coord = (coord + vect)*ANGS2BOHR
            str += "%6i %11.6f  %10.6f  %10.6f  %10.6f\n"%(atm, 0., shifted_coord[0],
                                                           shifted_coord[1], shifted_coord[2])

        grid /= avg 
        it = np.nditer(grid, flags=['multi_index'], order='C')
        while not it.finished:
            for i in range(6):
                str += " %11.6f"%it[0]
                b_before = it.multi_index[1]
                bool = it.iternext()
                if not bool or (it.multi_index[1] - b_before) != 0:
                    break
            str += "\n"
        return str
    
    def print_stats(self, rank):

        error = self.obtain_error(rank)
        adsorbophore = self.adsorbophore_from_sql(rank)
        vdw, el = [],[]
        for active_site in adsorbophore.active_sites:
            act_site = self.active_sites_db.get_active_site(active_site.name)
            vdw.append(act_site.vdweng)
            el.append(act_site.eleng)

        vdw = np.array(vdw)
        el = np.array(el)
        print "DATA for rank %i"%(rank)
        print "============================"
        print "adsorbophore size : %i"%(len(adsorbophore.active_sites))
        print "average VDW energy: %12.5f +/- %12.5f"%(np.mean(vdw), np.std(vdw))
        print "average ELE energy: %12.5f +/- %12.5f"%(np.mean(el), np.std(el))
        print "RMSD atoms        : %12.5f"%(error)

def rotation_from_vectors(v1, v2, point=None):
    """Obtain rotation matrix from sets of vectors.
    the original set is v1 and the vectors to rotate
    to are v2.

    """

    # v2 = transformed, v1 = neutral
    ua = np.array([np.mean(v1.T[0]), np.mean(v1.T[1]), np.mean(v1.T[2])])
    ub = np.array([np.mean(v2.T[0]), np.mean(v2.T[1]), np.mean(v2.T[2])])

    Covar = np.dot((v2 - ub).T, (v1 - ua))

    u, s, v = np.linalg.svd(Covar)
    uv = np.dot(u,v)
    d = np.identity(3) 
    d[2,2] = np.linalg.det(uv) # ensures non-reflected solution
    M = np.dot(np.dot(u,d), v)
    R = np.identity(4)
    R[:3,:3] = M
    if point is not None:
        R[:3,:3] = point - np.dot(M, point)
    return R

