#!/usr/bin/env python

import os
import sys
import pickle
import math
from optparse import OptionParser
import itertools
import networkx as nx
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle
from scipy.spatial import distance
from math import pi
from nets import Net
from sub_graphs import SubGraph
from correspondence_graphs import CorrGraph
from faps import Structure
from uuid import uuid4
global rank, size, comm
rank, size, comm = 0, 0, None
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    print "Warning! no module named mpi4py found, I hope you are not running this in parallel!"
    pass

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
        parser.add_option("--par", action="store_true",
                          default=False,
                          dest="MPI",
                          help="Toggle the parallel version of the code. Defalut serial.")
        (local_options, local_args) = parser.parse_args()
        self.options = local_options

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

class Tree(object):
    """Tree creates a pairwise iteration technique for pairing and narrowing
    down large datasets.  This will pair active sites, which will then
    be reduced to a pharmacophore, to be subsequently paired.
    
    """

    def __init__(self):
        """Include a plot of the tree to see which nodes got paired to which."""
        self.nodes = []
        self.names = []

    def add(self, name, val):
        self.names.append(name)
        self.nodes.append(val)

    def random_pairings(self, iterable):
        """Pair the elements in the list of nodes using the 
        Fisher-Yates algorithm.
        
        """
        l = [i for i in range(len(iterable)) if i%2 == 1] # odd
        q = [j for j in range(len(iterable)) if j%2 == 0] # even
        shuffle(l)
        shuffle(q)
        return itertools.izip_longest(l, q)

    def branchify(self, nodelist):
        if len(nodelist) > 1:
            newlist = list(self.random_pairings(nodelist))
            return newlist

    #def branchifyd(self):
    #    """recursive and exhaustive random pairing until the root node is reached."""
    #    # take all the nodes and names, pair them to yield new nodes and names,
    #    # then recurse until there is only one node/name.
    #    G = nx.DiGraph()
    #    for n1, n2 in self.random_pairings(self.nodes):
    #        if n1 is None:
    #            G.add_node(self.names[n2])
    #        elif n2 is None:
    #            G.add_node(self.names[n2])
    #        else:
    #            G.add_edge(self.names[n1], self.names[n2])

    #    F = nx.DiGraph()
    #    for u, v in G.edges():
    #        vals, distances = nx.bellman_ford(G, u)
    #        if min(distances.values()) < 2:
    #            for u, v in vals.items():
    #                if v:
    #                    F.add_edge(v, u)

    #    nx.draw(F)
    #    plt.show()


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
    def __init__(self, radii=4.5, min_atom_cutoff=9, max_pass_count=20, random_seed=None, tol=0.4):
        self.site_count = 0
        self._active_sites = {} # stores all subgraphs in a dictionary of lists.  The keys are the mof names
        self._radii = radii # radii to scan for active site atoms
        self._bad_pair = {} # store all binding site pairs which did not result in a pharmacophore
        self.max_pass_count = max_pass_count 
        self.min_atom_cutoff = min_atom_cutoff
        self.time = 0.
        self.tol = tol
        # initialize a random seed (for testing purposes)
        if random_seed is None:
            # make the seed time-based, but still be able to recover what it
            # is for tracking purposes
            self.seed = random.randint(0, sys.maxint)
        else:
            random.seed(random_seed)
            self.seed = random_seed
        # set the random seed
        random.seed(self.seed)

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

    def get_clique(self, g1, g2):
        """Returns the atoms in g1 which constitute a maximal clique
        with respect to g2 and a given tolerance.

        """
        cg = CorrGraph(sub_graph=g1)
        cg.pair_graph = g2
        cg.correspondence_api(tol=self.tol)
        mc = cg.extract_clique()
        clique = mc.next()
        return clique

    def failed_pair(self, nameset1, nameset2):
        for i,j in itertools.product(nameset1, nameset2):
            if tuple(sorted([i,j])) in self._bad_pair.keys():
                return True
        return False

    def get_bad_pairs(self, nameset1, nameset2):
        """assume that all individuals will not form a valid bond here."""
        bb = []
        for i,j in itertools.product(nameset1, nameset2):
            bad_pair = tuple(sorted([i,j]))
            bb.append(bad_pair)
        return bb

    def add_bad_pairs(self, pairs):
        for bad_pair in pairs:
            self._bad_pair.setdefault(bad_pair, 0)
            self._bad_pair[bad_pair] += 1

    def score(self):
        """Rank the subsequent binding sites based on their degree
        of diversity (metal count, organic count, fgroup count, number
        of sites included, number of MOFs included, average binding 
        energy, variation in binding energy.

        """
        pass

    def combine_pairs(self, pairings, nodes, names):
        """Combining pairs of active sites"""
        no_pairs = 0
        newnodes, newnames, new_bads = [], [], []
        for (i, j) in pairings:
            # check if the pair has already been tried, and failed.
            #if (i is not None)and(j is not None)and(
            #        self.failed_pair(names[i], names[j])):
            #    g1, g2 = nodes[i], nodes[j]
            #    newnodes.append(g1)
            #    newnodes.append(g2)
            #    newnames.append(names[i])
            #    newnames.append(names[j])
            #    no_pairs += 1
            #    continue
            # else: continue with the pharmacophore generation
            if i is not None and j is not None:
                g1, g2 = nodes[i], nodes[j]
                p = self.get_clique(g1, g2)
                # check if the clique is greater than the number of
                # atoms according to min_cutoff
                if len(p) >= self.min_atom_cutoff:
                    #(g1 % p).debug("Pharma")
                    newnodes.append( g1 % p )
                    newname = names[i] + names[j]
                    newnames.append(newname)
                
                # append to the bad_pair dictionary otherwise
                else:
                    #new_bads += self.get_bad_pairs(names[i], names[j])
                    newnodes.append(g1)
                    newnodes.append(g2)
                    newnames.append(names[i])
                    newnames.append(names[j])
                    no_pairs += 1
            # in cases where the number of active sites is odd,
            # the itertools function will produce a pair that has
            # (int, None). This is to account for that.
            # mpi routines will sometimes send (None, None)'s if 
            # the number of nodes is greater than the number of pairs
            # to compute.. 
            elif i is None and j is None:
                continue
            elif i is None:
                newnodes.append( nodes[j] )
                newnames.append( names[j] )
                no_pairs += 1
            elif j is None:
                newnodes.append( nodes[i] )
                newnames.append( names[i] )
                no_pairs += 1

        #print "rank %i"%rank, "pairings %i"%(len(pairings)), "no_pairs count %i"%no_pairs
        return newnodes, newnames, no_pairs, new_bads 

    def run_pharma_tree(self):
        """Take all active sites and join them randomly. This is a breadth
        first reverse-tree algorithm."""
        done = False
        t1 = time()
        tree = Tree()
        # collect all the active sites in nodes.. this is depreciated.
        for key, val in self._active_sites.items():
            for c, j in enumerate(val): 
                name = key + ".%i"%(c)
                tree.add(name, j)
        pairings = tree.branchify(tree.nodes) # initial pairing up of active sites
        nodes = tree.nodes   # initial arrays of nodes
        names = [[i] for i in tree.names] # initial list of lists of names
        pass_count = 0  # count the number of times the loop joins all bad pairs of active sites
        while not done:
            # loop over pairings, each successive pairing should narrow down the active sites
            newnodes, newnames, no_pairs, new_bad = self.combine_pairs(pairings, nodes, names)
            # this is a work-around for the mpi version, this was originally in the routine above
            # but now this dictionary needs to be broadcast to all nodes.
            self.add_bad_pairs(new_bad)
            # count the number of times no pairings were made due to 
            # all the entries being in the _bad_pair dictionary
            if no_pairs == len(pairings):
                pass_count += 1

            # TESTME(pboyd): This may really slow down the routine.
            else:
                # re-set if some good sites were found
                pass_count = 0
            # TESTME(pboyd)


            if len(newnodes) <= 1 or pass_count == self.max_pass_count:
                done = True

            else:
                nodes = newnodes[:]
                names = newnames[:]
                pairings = tree.branchify(range(len(nodes)))
        t2 = time()
        self.time = t2 - t1
        return newnodes, newnames

    def write_final_files(self, names, pharma_graphs): 
        pickle_dic = {}
        filebasename='%s_N.%i_r.%3.2f_ac.%i_ma.%i_rs.%i_t.%3.2f'%("pharma",
                                                 self.site_count,
                                                 self.radii,
                                                 self.min_atom_cutoff,
                                                 self.max_pass_count,
                                                 self.seed,
                                                 self.tol)
        for id, (name, pharma) in enumerate(zip(names, pharma_graphs)):
            pickle_dic[tuple(name)] = pharma
        f = open(filebasename+".pkl", 'wb')
        pickle.dump(pickle_dic, f)
        f.close()

class MPIPharmacophore(Pharmacophore):

    def chunks(self, l, n):
        """yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

    def run_pharma_tree(self):
        """Take all active sites and join them randomly. This is a breadth
        first reverse-tree algorithm."""
        t1 = time()
        done = False
        tree = Tree()
        # collect all the active sites in nodes.. this is depreciated.
        for key, val in self._active_sites.items():
            for c, j in enumerate(val):
                name = key + ".%i"%(c)
                tree.add(name, j)
        orig_pairings = tree.branchify(tree.nodes) # initial pairing up of active sites
        nodes = tree.nodes   # initial arrays of nodes
        names = [[i] for i in tree.names] # initial list of lists of names

        pass_count = 0  # count the number of times the loop joins all bad pairs of active sites
        while not done:
            # loop over pairings, each successive pairing should narrow down the active sites
            # broadcast the pairing to the nodes.
            chunks = None
            # Some MPI stuff
            if rank == 0:
                chunks = []
                sz = int(math.ceil(float(len(list(orig_pairings)))/ float(size)))
                if sz == 0:
                    sz = 1
                for i in self.chunks(orig_pairings, sz):
                    chunks.append(i)
                # shitty hack for making the last node do nothing
                for i in range(size-len(chunks)):
                    chunks.append([(None,None)])
            pairings = comm.scatter(chunks, root=0)
            newnodes, newnames, no_pairs, new_bad = self.combine_pairs(pairings, nodes, names)
            # make sure each node has the same self.bad_pairings
            new_bad = [i for j in comm.allgather(new_bad) for i in j]
            self.add_bad_pairs(new_bad)
            # collect the pairings.
            newnodes = comm.gather(newnodes, root=0)
            newnames = comm.gather(newnames, root=0)
            no_pairs = comm.gather(no_pairs, root=0)
            if rank == 0:
                no_pairs = sum(no_pairs)
                newnodes = [j for i in newnodes for j in i]
                newnames = [j for i in newnames for j in i]
                # for some reason the number of no_pairs can get larger than
                # the number of pairings broadcast to each node...?
                if no_pairs >= len(orig_pairings):
                    pass_count += 1
                # TESTME(pboyd): This may really slow down the routine.
                else:
                # re-set if some good sites were found
                    pass_count = 0

                # pass count was hard-coded to 9 for some reason.. I don't 
                # think it needs to be this high.
                if len(newnodes) <= 1 or pass_count == self.max_pass_count:
                    done = True

                else:
                    nodes = newnodes[:]
                    names = newnames[:]
                    orig_pairings = tree.branchify(range(len(nodes)))

            # broadcast the complete list of nodes and names to the other nodes.
            nodes = comm.bcast(newnodes, root=0)
            names = comm.bcast(newnames, root=0)
            done = comm.bcast(done, root=0)
        t2 = time()
        self.time = t2 - t1
        return newnodes, newnames

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

def site_xyz(dic):
    f = open("binding_site.xyz", "w")
    f.writelines("%i\n"%(len(dic.keys())))
    f.writelines("binding site\n")
    for key, val in dic.items():
        f.writelines("%s %9.4f %9.4f %9.4f\n"%(key[:1], val[0], val[1], val[2]))
    f.close()

def add_to_pharmacophore(mof, pharma, path):
    faps_graph = pharma.get_main_graph(mof)
    binding_sites = BindingSiteDiscovery(path)
    if binding_sites.absl_calc_exists():
        binding_sites.co2_xyz_parser()
        binding_sites.energy_parser()
        for site in binding_sites.sites:
            el = site.pop('electrostatic')
            vdw = site.pop('vanderwaals')
            if el + vdw < -30.:
                coords = np.array([site[i] for i in ['C', 'O1', 'O2']])
                active_site = pharma.get_active_site(coords, faps_graph)
                # set all elements to C so that the graph matching is non-discriminatory
                #active_site.set_elem_to_C()
                pharma.store_active_site(active_site, name=mof.name)
                #if count == 2:
                #    shift = site['C']
                #    site['C'] = site['C'] - shift
                #    site['O1'] = site['O1'] - shift
                #    site['O2'] = site['O2'] - shift
                #    #site_xyz(site)
                #    #active_site.debug("active_site")


def main_pharma():
    options = CommandLine().options
    t1 = time()

    _MOF_STRING = "/share/scratch/pboyd/binding_sites"
    if options.MPI:
        pharma = MPIPharmacophore(
                           radii=options.radii,
                           min_atom_cutoff=options.mac,
                           max_pass_count=options.num_pass,
                           random_seed=options.seed,
                           tol=options.tol)
    else:
        pharma = Pharmacophore(
                           radii=options.radii,
                           min_atom_cutoff=options.mac,
                           max_pass_count=options.num_pass,
                           random_seed=options.seed,
                           tol=options.tol)
    directory = MOFDiscovery(_MOF_STRING)
    count = 0
    for mof, path in directory.dir_scan():
        if count == options.nmofs:
            break
        faps_mof = Structure(name=mof)
        faps_mof.from_cif(os.path.join(path, mof+".cif"))
        add_to_pharmacophore(faps_mof, pharma, path)
        count+=1
    t2 = time()
    #if rank==0:
    #    print "number of binding sites = %i"%(pharma.site_count)
    #sys.exit()
    final_nodes, final_names = pharma.run_pharma_tree()
    t3 = time()
    if rank == 0:
        # write the pickle stuff
        pharma.write_final_files(final_names, final_nodes)
        print "Finished. Scanned %i binding sites"%(pharma.site_count)
        #print "   time for initialization = %5.3f seconds"%(t2-t1)
        print "time for pharma discovery = %5.3f seconds"%(pharma.time)
        #print "                Total time = %5.3f seconds"%(t3-t1)

if __name__ == "__main__":
    main_pharma()
    #main()

