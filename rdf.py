#!/usr/bin/env python
import sys
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
sys.path[:0] = '/home/pboyd/codes_in_development/net_discovery'

from sqlbackend import Net_sql, SQLNet

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
                float(nconfigs)

    def _determine_supercell(self):
        """Currently set to 3x3x3. Determine the supercell that would
        fit a box of size self.max_dist"""
        return (-1, 0, 1)

    def _radii(self, i):
        return self.dr*i

    def compute_rdf(self, label, key="C"):
        hist = self._get_histogram(label, key=key)
        norm = self._get_norm(label)
        rdf = [hist[i]/norm/(self._radii(i)*self._radii(i) + self.dr*self.dr/12.)
                for i in range(self.bins)]
        return rdf

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
            self.update({net.name: nn})

def write_rdf_file(filename, histogram, dr):
    f = open(filename, 'w')
    print "Writing %s"%(filename)
    f.writelines("r,g(r)\n")
    for i, val in enumerate(histogram):
        f.writelines("%6.4f,%9.6f\n"%(i*dr, val))
    f.close()



def main():
    #_NET_STRING = "/home/pboyd/co2_screening/final_MOFs/top/net_analysis/net_top.pkl"
    _NET_STRING = "/share/scratch/pboyd/co2_screening/final_MOFs/mmol_g_above2/net_scan/mmol_g_above2.db"
    _MOF_STRING = "/share/scratch/pboyd/binding_sites/"
    RDF = PairDistFn(bins=20)
    RDF.init_co2_arrays()
    nets = NetLoader(_NET_STRING)
    directory = MOFDiscovery(_MOF_STRING)
    directory.dir_scan()
    for mof, path in directory.mof_dirs:
        binding_sites = BindingSiteDiscovery(path)
        if binding_sites.absl_calc_exists():
            binding_sites.co2_xyz_parser()
            binding_sites.energy_parser()
            try:
                RDF.bin_co2_distr(binding_sites.sites, nets[mof], energy_cut=30.)
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
