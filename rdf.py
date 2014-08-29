#!/usr/bin/env python
import itertools
import sys
import os
import numpy as np
from numpy import pi
from scipy.spatial import distance
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
sys.path[:0] = '/home/pboyd/codes_in_development/net_discovery'

from sqlbackend import Net_sql, SQLNet

Base = declarative_base()

class BindingSiteDiscovery(object):
    """Looks for binding site analysis, searches for check_dlpoly.out,
    associates binding site.? to the order in check_dlpoly.out. Thus
    matching energies.

    ***Currently calibrated for CO2 runs only***
   
    """

    def __init__(self, mofdir, temp=298.0, press=0.15):
        self.state_temp = temp
        self.state_press = press
        self.mofdir = mofdir
        self.mofname = os.path.basename(mofdir)
        self.sites = []
        self.absl_dir = self.mofname + "_absl"
        self.energy_file = "check_dlpoly.out"
        self.coord_file = "absl_post.0.025.xyz" # Not sure if the decimal number will change 

    def absl_calc_exists(self):
        """Determine if the absl program was run on this MOF. At the
        prescribed temperature and pressure."""
        faps_fastmc_basedir = "faps_%s_fastmc"%(self.mofname)
        faps_gcmc_run = "T%.1f"%(self.state_temp) + "P%.2f"%(self.state_press)
        return \
        os.path.isdir(os.path.join(self.mofdir, 
                                   faps_fastmc_basedir, 
                                   faps_gcmc_run)) and \
        os.path.isfile(os.path.join(self.mofdir,
                                    self.absl_dir,
                                    self.energy_file)) and \
        os.path.isfile(os.path.join(self.mofdir,
                                    self.absl_dir,
                                    self.coord_file))

    def co2_xyz_parser(self):
        """Parses the xyz file into separate molecules. The order should be 
        related to the order found in the check_dlpoly.out file.
        
        """
        n_co2 = 3
        xyz = self._read_xyz(os.path.join(self.mofdir,
                                          self.absl_dir,
                                          self.coord_file))
        natoms = xyz.next()
        self.nmol = natoms / n_co2
        if natoms % n_co2 != 0:
            print "warning('number of atoms does not coincide with co2!')"
        #print "debug('number of binding sites = %i')"%self.nmol

        for mol in range(self.nmol):
            self.sites.append({})
            for i in range(n_co2):
                atom, coord, idk1, idk2 = xyz.next()
                if atom == "O":
                    at = atom+"1"
                else:
                    at = atom
                if self.sites[-1].has_key(at):
                    at = atom + "2"
                self.sites[-1][at] = np.array(coord)

    def _read_xyz(self, filename):
        with open(filename) as f:
            #burn first two
            yield int(f.readline().strip())
            f.readline()
            for line in f:
                pl = line.strip()
                pl = pl.split()
                yield (pl[0], [float(i) for i in pl[1:4]], 
                        float(pl[4]), int(pl[5]))

    def energy_parser(self):
        energy_file = os.path.join(self.mofdir,
                                   self.absl_dir,
                                   self.energy_file)

        read = False
        site = 0
        for file_line in open(energy_file):
            file_line = file_line.strip()
            if file_line and read:
                parsed_line = file_line.split()
                self.sites[site]['electrostatic'] = float(parsed_line[4])
                self.sites[site]['vanderwaals'] = float(parsed_line[3])
                site += 1
            if file_line == '# site total binding (KJ/mol) noncovalent (kJ/mol) elec (KJ/mol)':
                read = True

class PairDistFn(object):
    """Creates all radial distributions from a set of binding sites to
    their functional groups.

    """
    def __init__(self, bins=10, max=10):
        self.label_counts = {}
        self.bins = bins
        self.max_dist = max
        self.vol_av = []

    def init_co2_arrays(self):
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

class MOFDiscovery(object):
    def __init__(self, directory):
        """Searches a directory for MOFs.  MOFs in this sense are defined
        by the string mof_def. 
        
        """
        self.mof_def = "str_"
        self.directory = directory
        self.mof_dirs = []

    def dir_scan(self, max_mofs=None):
        """Makes assumption that the mof name is the base directory."""
        count = 0 
        f = open("MOFs_used.csv", "w")
        for root, directories, filenames in os.walk(self.directory):
            mofdir = os.path.basename(root)
            if self._valid_mofdir(mofdir):
                self.mof_dirs.append((mofdir,root))
                f.writelines("%s\n"%mofdir)
                count += 1
            if (max_mofs is not None) and (count == max_mofs):
                return
        f.close()
    
    def _valid_mofdir(self, dir):
        """Returns True if the directory describes the base directory of a 
        faps/absl run. Currently the criteria is that it starts with str_
        and does not end with _absl.  May need to be more robust in the 
        future.

        """
        return dir.startswith(self.mof_def) and not dir.endswith("_absl")

class NetLoader(dict):
    def __init__(self, sql_file):
        self.engine = create_engine('sqlite:///%s'%(sql_file))
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self._read_sql()
    def _read_sql(self):
        for net in self.session.query(SQLNet).all():
            nn = Net_sql(net.mofname)
            nn.from_database(net)
            self.update({net.mofname: nn})

def write_rdf_file(filename, histogram, dr):
    f = open(filename, 'w')
    print "Writing %s"%(filename)
    f.writelines("r,g(r)\n")
    for i, val in enumerate(histogram):
        f.writelines("%6.4f,%9.6f\n"%(i*dr, val))
    f.close()



def main(_NET_DB, _MOF_DIR, NUM_BINS, ENCUT):
    RDF = PairDistFn(bins=NUM_BINS)
    RDF.init_co2_arrays()
    nets = NetLoader(_NET_DB)
    directory = MOFDiscovery(_MOF_DIR)
    directory.dir_scan()
    print "Finished reading in files, starting RDF calc"
    for mof, path in directory.mof_dirs:
        binding_sites = BindingSiteDiscovery(path)
        if binding_sites.absl_calc_exists():
            binding_sites.co2_xyz_parser()
            binding_sites.energy_parser()
            try:
                RDF.bin_co2_distr(binding_sites.sites, nets[mof], energy_cut=ENCUT)
            except KeyError:
                pass
                #print "Could not find the net for mof %s"%mof
        else: 
            print "warning('%s could not be found!')"%mof

    for label in RDF.label_counts.keys():
        hist = RDF.compute_rdf(label, key="C")
        filename="%s_rdf.%s.csv"%(label, "C")
        write_rdf_file(filename, hist, RDF.dr)
        hist = RDF.compute_rdf(label, key="O")
        filename="%s_rdf.%s.csv"%(label, "O")
        write_rdf_file(filename, hist, RDF.dr)

if __name__=="__main__":
    nets = "/share/scratch/pboyd/co2_screening/final_MOFs/mmol_g_above2/net_scan/mmol_g_above2.db"
    #nets = "/share/scratch/pboyd/co2_screening/final_MOFs/mmol_g_above2/net_scan/xab_dir/xab.job.db"
    mofs = "/share/scratch/pboyd/binding_sites/"
    #mofs = "/share/scratch/ekadants/top_REPEAT/"
    main(nets, mofs, 60, 30.)
