import numpy as np
from config_fap import Options
from StringIO import StringIO
from faps import Structure, Atom, Cell, mk_gcmc_control, Guest
import subprocess
from sql_backend import Data_Storage, SQL_Pharma, SQL_ActiveSite, SQL_ActiveSiteAtoms, SQL_Distances, SQL_ActiveSiteCO2
from sql_backend import SQL_Adsorbophore, SQL_AdsorbophoreSite, SQL_AdsorbophoreSiteIndices
from math import pi
import os

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

    def add_fragment(self, net_obj, lattice):
        cell = Cell()
        cell._cell = np.array(lattice)
        self.struct.cell = cell
        for id, (elem, posi) in enumerate(zip(net_obj.elements, net_obj._coordinates)):
            atom = Atom(at_type=elem, pos=posi, parent=self.struct, charge=net_obj.charges[id])
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

class Something(object):
    
    def obtain_error(self, name, sites, debug_ind=0):
        if not isinstance(name, tuple):
            return 0.
        #site, qq, qq, qq = self.get_active_site_from_sql(name[0])
        try:
            site = self._active_sites[name[0]]
        except KeyError:
            site = self.get_active_site_graph_from_sql(name[0])
        base = site % [j[0] for j in sites]
        base.shift_by_centre_of_atoms()
        mean_errors = []
        for ii in range(1, len(name)):
            atms = [q[ii] for q in sites]
            #site, qq, qq, qq = self.get_active_site_from_sql(name[ii])
            try:
                site = self._active_sites[name[ii]]
            except KeyError:
                site = self.get_active_site_graph_from_sql(name[ii])
            match = site % atms
            T = match.centre_of_atoms.copy()
            match.shift_by_centre_of_atoms()
            R = rotation_from_vectors(match._coordinates[:], 
                                      base._coordinates[:])
            #R = rotation_from_vectors(base._coordinates[:],
            #                          match._coordinates[:]) 
            match.rotate(R)
            #co2 = self._co2_sites[name[ii]]
            #co2 -= T
            #co2 = np.dot(R[:3,:3],co2.T).T
            #match.debug('clique%i'%debug_ind)
            #f = open('clique%i.xyz'%debug_ind, 'a')
            #for at, (x, y, z) in zip(['C', 'O', 'O'], co2):
            #    f.writelines("%s %12.5f %12.5f %12.5f\n"%(at, x, y, z))
            #f.close()
            #NB THIS MIGHT BE BIG.
            #mean_errors.append(np.sqrt(np.mean((base._coordinates - match._coordinates)**2)))
            for p, q in zip(base._coordinates, match._coordinates):
                mean_errors.append((p-q)**2)
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
    
    def obtain_co2_fragment_energies(self, name, site, co2, i=0):
        #nn = name[id]
        #ss = [j[id] for j in sites]
        mofname = '.'.join(name.split('.')[:-1])
        cell = (self.lattices[mofname].T * np.array([5.,5.,5.])).T
        fastmc = Fastmc_run(supercell=(1,1,1))
        fastmc.add_guest(co2)
        fastmc.add_fragment(site, cell)
        fastmc.run_fastmc()
        vdw, el = fastmc.obtain_energy()
        fastmc.clean(i)
        return vdw*4.184, el*4.184

    def obtain_co2_distribution(self, name, sites, ngridx=70, ngridy=70, ngridz=70):

        gridc = np.zeros((ngridx, ngridy, ngridz))
        grido = np.zeros((ngridx, ngridy, ngridz))

        #gridevdw = np.zeros((ngridx, ngridy, ngridz))
        #grideel = np.zeros((ngridx, ngridy, ngridz))
        gride = np.zeros((ngridx, ngridy, ngridz))
        #gridevdwcount = np.ones((ngridx, ngridy, ngridz))
        #grideelcount = np.ones((ngridx, ngridy, ngridz))
        gridecount = np.ones((ngridx, ngridy, ngridz))
        if not isinstance(name, tuple):
            return None, None
        _2radii = self.radii*2. + 2.
        # Because the cube file puts the pharmacophore in the middle of the box,
        # we need to shift the CO2 distributions to centre at the middle of the box
        # this was originally set to the radial cutoff distance of the initial
        # pharmacophore 
        shift_vector = np.array([_2radii/2., _2radii/2., _2radii/2.])
        #shift_vector = np.array([self._radii, self._radii, self._radii])
        nx, ny, nz = _2radii/float(ngridx), _2radii/float(ngridy), _2radii/float(ngridz)
        co2, vdweng, eleng = self.get_active_site_from_sql(name[0])
        try:
            site = self._active_sites[name[0]]
        except KeyError:
            site = self.get_active_site_graph_from_sql(name[0])
        #co2 = self._co2_sites[name[0]]
        base = site % [j[0] for j in sites]
        T = base.centre_of_atoms[:3].copy()
        base.shift_by_centre_of_atoms()
        inds = self.get_grid_indices(co2 - T + shift_vector, (nx,ny,nz))
        self.increment_grid(gridc, inds[0])
        self.increment_grid(grido, inds[1:])
        evdw, eel = self.obtain_co2_fragment_energies(name[0], base, co2-T, 0)
        #evdw /= vdweng
        #eel /= eleng
        #self.increment_grid(gridevdw, inds[0], en=evdw)
        #self.increment_grid(grideel, inds[0], en=eel)
        self.increment_grid(gride, inds[0], en=eel+evdw)

        for ii in range(1, len(name)):

            atms = [q[ii] for q in sites]
            co2, vdweng, eleng = self.get_active_site_from_sql(name[ii])
            try:
                site = self._active_sites[name[ii]]
            except KeyError:
                site = self.get_active_site_graph_from_sql(name[ii])
            match = site % atms
            T = match.centre_of_atoms[:3].copy()
            match.shift_by_centre_of_atoms()
            R = rotation_from_vectors(match._coordinates[:], 
                                      base._coordinates[:])
            #R = rotation_from_vectors(base._coordinates[:], 
            #                          match._coordinates[:])
            match.rotate(R)
            #co2 = self._co2_sites[name[ii]].copy()
            co2 -= T
            co2 = np.dot(R[:3,:3], co2.T).T
            inds = self.get_grid_indices((co2+shift_vector), (nx,ny,nz))
            self.increment_grid(gridc, inds[0])
            self.increment_grid(grido, inds[1:])
            evdw, eel = self.obtain_co2_fragment_energies(name[ii], match, co2, ii)
            #evdw /= vdweng
            #eel /= eleng
            #self.increment_grid(gridevdw, inds[0], en=evdw)
            #self.increment_grid(grideel, inds[0], en=eel)
            self.increment_grid(gride, inds[0], en=eel+evdw)
            #self.increment_grid(gridevdwcount, inds[0])
            #self.increment_grid(grideelcount, inds[0]) 
            self.increment_grid(gridecount, inds[0]) 
            # bin the x, y, and z
        string_gridc = self.get_cube_format(base, gridc, len(name), float(len(name)),
                                                ngridx, ngridy, ngridz)
        string_grido = self.get_cube_format(base, grido, len(name), float(len(name)), 
                                                ngridx, ngridy, ngridz)
        #string_gridevdw = self.get_cube_format(base, gridevdw, len(name), gridevdwcount,
        #                                        ngridx, ngridy, ngridz)
        #string_grideel = self.get_cube_format(base, grideel, len(name), grideelcount,
        #                                        ngridx, ngridy, ngridz)
        string_gride = self.get_cube_format(base, gride, len(name), gridecount,
                                            ngridx, ngridy, ngridz)
        return string_gridc, string_grido, string_gride #string_gridevdw, string_grideel

    def get_cube_format(self, clique, grid, count, avg, ngridx, ngridy, ngridz):
        # header
        str = "Clique containing %i binding sites\n"%count
        str += "outer loop a, middle loop b, inner loop c\n"
        str += "%6i %11.6f %11.6f %11.6f\n"%(len(clique), 0., 0., 0.)
        _2radii = (self._radii*2. + 2.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, _2radii*ANGS2BOHR/float(ngridx), 0., 0.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, 0., _2radii*ANGS2BOHR/float(ngridy), 0.)
        str += "%6i %11.6f %11.6f %11.6f\n"%(ngridx, 0., 0., _2radii*ANGS2BOHR/float(ngridz))
        vect = np.array([_2radii/2., _2radii/2., _2radii/2.]) 
        for i in range(len(clique)):
            atm = ATOMIC_NUMBER.index(clique.elements[i])
            coords = (clique._coordinates[i] + vect)*ANGS2BOHR
            str += "%6i %11.6f  %10.6f  %10.6f  %10.6f\n"%(atm, 0., coords[0],
                                                           coords[1], coords[2])

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

    def write_final_files(self, names, pharma_graphs, site_count):
        #pickle_dic = {}
        #co2_pos_dic = {}
        #co2_dist_dic = {}
        filebasename='%s_N.%i_r.%3.2f_ac.%i_ma.%i_rs.%i_t.%3.2f'%("pharma",
                                                 site_count,
                                                 self.radii,
                                                 self.min_atom_cutoff,
                                                 self.max_pass_count,
                                                 self.seed,
                                                 self.tol)
        #f = open(filebasename+".csv", 'w')
        data_storage = Data_Storage(filebasename)
        ranking = []
        datas = {}
        #f.writelines("length,pharma_std,el_avg,el_std,vdw_avg,vdw_std,tot_avg,tot_std,name\n") 
        for id, (name, pharma) in enumerate(zip(names, pharma_graphs)):
            
            el, vdw, tot = [], [], []
            if isinstance(name, tuple):
                for n in name:
                    co2, vdw_en, el_en = self.get_active_site_from_sql(n)

                    el.append(el_en)
                    vdw.append(vdw_en)
                    tot.append(el_en+vdw_en)

                elavg = np.mean(el)
                elstd = np.std(el)
                vdwavg = np.mean(vdw)
                vdwstd = np.std(vdw)
                totavg = np.mean(tot)
                totstd = np.std(tot)
                pharma_length = len(name)
            else:
                pharma_length = 1
                co2, vdw_en, el_en = self.get_active_site_from_sql(name)
                
                elavg = el_en 
                elstd = 0. 
                vdwavg = vdw_en 
                vdwstd = 0. 
                totavg = el_en + vdw_en 
                totstd = 0.
            sql = SQL_Pharma(str(name), vdwavg, vdwstd, elavg, elstd, pharma_length)
            ranking.append((pharma_length, str(name)))
            # only store the co2 distributions from the more important binding sites, currently set to
            # 0.1 % of the number of original binding sites
            error = self.obtain_error(name, pharma, debug_ind=id)
            error = 0.
            sql.set_error(error)
            if isinstance(name, tuple) and pharma_length >= int(site_count*0.001):
                # store the nets, this is depreciated..
                #site.shift_by_centre_of_atoms()
                #pickle_dic[name] = site 
                # store the CO2 distributions
                cdist, odist, edist = self.obtain_co2_distribution(name, pharma)
                sql.set_probs(cdist, odist, edist)
                #co2_dist_dic[name] = (cdist, odist)
                # compute the energy distribution from the extracted site (function of radii)

            datas[str(name)] = sql
            #f.writelines("%i,%f,%f,%f,%f,%f,%f,%f,%s\n"%(len(name),error,elavg,elstd,vdwavg,
            #                                             vdwstd,totavg,totstd,name))

        for rank, (length, name) in enumerate(reversed(sorted(ranking))):
            sqlr = datas[name]
            sqlr.set_rank(rank)
            data_storage.store(sqlr)
        data_storage.flush()
        #f.close()
        #f = open(filebasename+".pkl", 'wb')
        #pickle.dump(pickle_dic, f)
        #f.close()
        #f = open(filebasename+"_co2dist.pkl", 'wb')
        #pickle.dump(co2_dist_dic, f)
        #f.close()
