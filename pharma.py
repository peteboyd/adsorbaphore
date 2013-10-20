#!/usr/bin/env python

import os
import sys
import pickle
import itertools
import numpy as np
from matplotlib import pyplot
from scipy.spatial import distance
from math import pi
from nets import Net
from sub_graphs import SubGraph
from correspondence_graphs import CorrGraph
from faps import Structure
from uuid import uuid4

"""
Designed to compute RDFs.  Read in binding site xyz, and associated energy.  Compute distances to functional groups.
Plot RDFs
"""


class Plot(object):
    def __init__(self):
        pass

class MOFDiscovery(object):
    def __init__(self, directory):
        """Searches a directory for MOFs.  MOFs in this sense are defined
        by the string mof_def. 
        
        """
        self.mof_def = "str_"
        self.directory = directory

    def dir_scan(self):
        """Makes assumption that the mof name is the base directory."""
        for root, directories, filenames in os.walk(self.directory):
            mofdir = os.path.basename(root)
            if self._valid_mofdir(mofdir):
                yield (mofdir, root)

    def _valid_mofdir(self, dir):
        """Returns True if the directory describes the base directory of a 
        faps/absl run. Currently the criteria is that it starts with str_
        and does not end with _absl.  May need to be more robust in the 
        future.

        """
        return dir.startswith(self.mof_def) and not dir.endswith("_absl")

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

class NetLoader(dict):
    def __init__(self, pickle_file):
        self._read_pickle(pickle_file)

    def _read_pickle(self, file):
        f = open(file, 'rb')
        self.update(pickle.load(f))

class Pharmacophore(object):

    """Grab atoms surrounding binding sites. compare with maximal clique detection."""
    def __init__(self, radii=3.5):
        self.site_count = 0
        self._active_sites = {} # stores all subgraphs in a dictionary of lists.  The keys are the mof names
        self._radii = radii # radii to scan for active site atoms
        self.main_pharma = None

    @property
    def radii(self):
        return self._radii
    @radii.setter
    def radii(self, val):
        self._radii = val

    def get_main_graph(self, faps_obj):
        """Takes a structure object and converts to a sub_graph using
        routines borrowed from net_finder."""
        s = SubGraph(name=faps_obj.name)
        # in the future, determine the size of the supercell which would fit the radii property
        s.from_faps(faps_obj, supercell=(2,2,2))
        return s
    
    def get_active_site(self, binding_site, subgraph):
        """Takes all atoms within a radii of the atoms in the binding site."""
        distances = distance.cdist(binding_site, subgraph._coordinates)
        sub_idx = []
        for (x, y), val in np.ndenumerate(distances):
            if val < self.radii:
                sub_idx.append(y)
        sub_idx = list(set(sub_idx))
        site = subgraph % sub_idx
        site.shift_by_vector(binding_site[0])
        return site 

    def store_active_site(self, activesite, name='Default'):
        self.site_count += 1
        self._active_sites.setdefault(name, []).append(activesite)

    def run_pharma(self):
        """Run graph recognition algorithm to find the clique."""
        actives, new_cliques, track_o_graph = [], [], []
        for key, val in self._active_sites.items():
            actives += val
        # create all pairs to compare active sites
        pairs = itertools.combinations(range(self.site_count), 2)
        print "size of active sites", len(actives)
        # PETE - A ROUND ROBIN HERE WHERE ALL PAIRS ARE THEN PAIRED ETC, UNTIL ALL PHARMA
        # ARE FOUND, NEED TO CATCH ONES WHICH DON'T MAKE IT TO THE END OF THE LOOP
        # DOES THIS GUARANTEE ALL CLIQUES > 4?
        # I DON'T THINK A ROUND-ROBIN IS NECESSARY, JUST ONE LOOP SHOULD DO?
        for outter in range(len(actives)):
            g1 = actives[outter]
            for inner in range(outter+1, len(actives)):
                g2 = actives[inner]
                cg = CorrGraph(sub_graph=g1)
                cg.pair_graph = g2
                cg.correspondence_api(tol=0.4)
                mc = cg.extract_clique()
                clique = mc.next()
                if clique > 4:
                    g1 = g1 % clique
                    print g1.elements
                    print g1._coordinates
                else:
                    cg = CorrGraph(sub_graph=actives[outter])
                    cg.pair_graph = g2
                    cg.correspondence_api(tol=0.4)
                    mc = cg.extract_clique()
                    clique = mc.next()
                    if clique > 4:
                        g1 = g1 % clique
                    else:
                        pass
            sys.exit()


        while not done:
            try:
                i1, i2 = pairs.next()
                g1, g2 = actives[i1], actives[i2]
                cg = CorrGraph(sub_graph=g1)
                cg.pair_graph = g2
                cg.correspondence_api(tol=0.01)
                mc = cg.extract_clique()
                # just take the first hit
                clique = mc.next()
                if len(clique) > 3:
                    sub_g = g1 % clique
                    new_cliques.append(sub_g)
                else:
                    if i1 not in track_o_graph:
                        new_cliques.append(g1)
                        track_o_graph.append(i1)
                    if i2 not in track_o_graph:
                        new_cliques.append(g2)
                        track_o_graph.append(i2)

            except StopIteration:
                if not new_cliques:
                    done = True
                else:
                    actives = new_cliques
                    print "size of active sites", len(actives)
                    new_cliques = []
                    track_o_graph = []
                    pairs = itertools.combinations(range(len(actives)), 2)




class PairDistFn(object):
    """Creates all radial distributions from a set of binding sites to
    their functional groups.

    """
    def __init__(self, bins=10, max=10):
        self.label_counts = {}
        self.bins = bins
        self.max_dist = max

    def init_co2_arrays(self):
        self._C_distances = {}
        self._O_distances = {}

    def bin_co2_distr(self, sites, net, energy_cut=None):
        """take the current MOF and obtain the radial distributions."""
        cells = self._determine_supercell()
        cell = net.cell
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
        self.rho = 1.
        nconfigs = self.label_counts[label]['nconfig']
        nparticles = self.label_counts[label]['nparticle']
        return 2. * pi * self.dr * self.rho * float(nconfigs) * \
                float(nparticles)/float(nconfigs)

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

def write_rdf_file(filename, histogram, dr):
    f = open(filename, 'w')
    print "Writing %s"%(filename)
    f.writelines("r,g(r)\n")
    for i, val in enumerate(histogram):
        f.writelines("%6.4f,%9.6f\n"%(i*dr, val))
    f.close()

def main():
    #_NET_STRING = "/home/pboyd/co2_screening/final_MOFs/top/net_analysis/net_top.pkl"
    _NET_STRING = "nets.pkl"
    _MOF_STRING = "./"
    RDF = PairDistFn(bins=20)
    RDF.init_co2_arrays()
    nets = NetLoader(_NET_STRING)
    directory = MOFDiscovery(_MOF_STRING)
    for mof, path in directory.dir_scan():
        #print mof
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

def main_pharma():
    _MOF_STRING = "./"
    pharma = Pharmacophore()
    pharma.radii = 4.5 
    directory = MOFDiscovery(_MOF_STRING)
    for mof, path in directory.dir_scan():
        print mof
        faps_mof = Structure(name=mof)
        faps_mof.from_cif(os.path.join(path, mof+".cif"))
        faps_graph = pharma.get_main_graph(faps_mof)

        binding_sites = BindingSiteDiscovery(path)
        if binding_sites.absl_calc_exists():
            binding_sites.co2_xyz_parser()
            binding_sites.energy_parser()
            for site in binding_sites.sites:
                if site['electrostatic'] + site['vanderwaals'] < -30.:
                    coords = np.array([site[i] for i in ['C', 'O1', 'O2']])
                    active_site = pharma.get_active_site(coords, faps_graph)
                    # set all elements to C so that the graph matching is non-discriminatory
                    #active_site.set_elem_to_C()
                    pharma.store_active_site(active_site, name=mof)
        break
    pharma.run_pharma()
    print "number of sites to evaluate", pharma.site_count
if __name__ == "__main__":
    main_pharma()
    #main()

