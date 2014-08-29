import numpy as np
from config_fap import Options
from StringIO import StringIO
from faps import Structure, Atom, Cell, mk_gcmc_control, Guest

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

