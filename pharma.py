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
#from correspondence_graphs import CorrGraph
sys.path.append("/home/pboyd/codes_in_development/mcqd_api/build/lib.linux-x86_64-2.7")
sys.path.append("/home/pboyd/codes_in_development/topcryst")
from LinAlg import rotation_from_vectors
import _mcqd as mcqd

from faps import Structure
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

class MPITools(object):
    
    def chunks(self, l, n):
        """yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i+n]
    
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
        parser.add_option("--mpi", action="store_true",
                          default=False,
                          dest="MPI",
                          help="Toggle the parallel version of the code. Default serial.")
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

#class MPIMOFDiscovery(MOFDiscovery, MPITools):
#    
#    def dir_scan(self, max_mofs=None):
#        chunks, ranks = None, None
#        if rank == 0:
#            f = open("MOFs_used.csv", "w")
#            count = 0 
#            for root, directories, filenames in os.walk(self.directory):
#                mofdir = os.path.basename(root)
#                if self._valid_mofdir(mofdir):
#                    f.writelines("%s\n"%mofdir)
#                    self.mof_dirs.append((mofdir,root))
#                    count += 1
#                if (max_mofs is not None) and (count == max_mofs):
#                    break
#            f.close()
#            sz = int(math.ceil(float(len(self.mof_dirs)) / float(size)))
#            ranks, chunks = [], []
#            for rr, ch in enumerate(self.chunks(self.mof_dirs, sz)):
#                ranks += [rr for i in range(len(ch))]
#                chunks.append(ch)
#        self.mof_dirs = comm.scatter(chunks, root=0)

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
        self.el_energy = {}
        self.vdw_energy = {}
        self._active_sites = {} # stores all subgraphs in a dictionary of lists.  The keys are the mof names
        self._co2_sites = {}
        self._radii = radii # radii to scan for active site atoms
        self._bad_pair = {} # store all binding site pairs which did not result in a pharmacophore
        self.max_pass_count = max_pass_count 
        self.min_atom_cutoff = min_atom_cutoff
        self.time = 0.
        self.tol = tol
        self.grid = (10, 10, 10)
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

    def store_active_site(self, activesite, name='Default', el_energy=0., vdw_energy=0.):
        self.site_count += 1
        self.el_energy[name] = el_energy
        self.vdw_energy[name] = vdw_energy
        self._active_sites[name] = activesite

    def store_co2_pos(self, pos, name='Default'):
        self._co2_sites[name] = pos

    def get_clique(self, g1, g2):
        """Returns the atoms in g1 which constitute a maximal clique
        with respect to g2 and a given tolerance.

        """
        nodes = mcqd.correspondence(g1.elements, g2.elements)
        if not nodes:
            return []
        adj_matrix = mcqd.correspondence_edges(nodes,
                                               g1.distances,
                                               g2.distances,
                                               self.tol)
        size = len(nodes)
        clique = mcqd.maxclique(adj_matrix, size)
        #cg.correspondence_api(tol=self.tol)
        #cg.correspondence(tol=self.tol)
        #mc = cg.extract_clique()
        #clique = mc.next()
        #g = min(g1, g2, key=len)
        #clique = range(len(g))
        if not clique:
            return [],[]
        else:
            return [nodes[i][0] for i in clique], [nodes[i][1] for i in clique]

    def failed_pair(self, nameset1, nameset2):
        for i,j in itertools.product(nameset1, nameset2):
            if tuple(sorted([i,j])) in self._bad_pair.keys():
                return True
        return False

    def score(self):
        """Rank the subsequent binding sites based on their degree
        of diversity (metal count, organic count, fgroup count, number
        of sites included, number of MOFs included, average binding 
        energy, variation in binding energy.

        """
        pass

    def get_rep_site_name(self, n):
        if isinstance(n, tuple):
            return n[0]
        elif isinstance(n, str):
            return n

    def get_rep_nodes(self, name, sites):
        return [i if isinstance(i, int) else i[0] for i in sites[name]]

    def combine_pairs(self, pairings, sites):
        """Combining pairs of active sites"""
        no_pairs = 0
        for (i, j) in pairings:
            if i is not None and j is not None:
                # get the representative names, representative atoms
                namei, namej = self.get_rep_site_name(i), self.get_rep_site_name(j)
                asi, asj = self._active_sites[namei], self._active_sites[namej]
                nodesi, nodesj = self.get_rep_nodes(i, sites), self.get_rep_nodes(j, sites)
                g1, g2 = (asi % nodesi), (asj % nodesj)
                p,q = self.get_clique(g1, g2)
                # check if the clique is greater than the number of
                # atoms according to min_cutoff
                if len(p) >= self.min_atom_cutoff:
                    #(g1 % p).debug("Pharma")
                    newnode = self.create_node_from_pair([sites[i][xx] for xx in p], 
                                                         [sites[j][yy] for yy in q])
                    newname = self.create_name_from_pair(i, j)
                    sites.update({newname:newnode})
                    del sites[i]
                    del sites[j]
                # append to the bad_pair dictionary otherwise
                else:
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
                no_pairs += 1
            elif j is None:
                no_pairs += 1
        return no_pairs

    def create_node_from_pair(self, arg1, arg2):
        sum = []
        for ii, jj in zip(arg1, arg2):
            instsum = []
            if isinstance(ii, tuple):
                for i in ii:
                    instsum.append(i)
            elif isinstance(ii, int):
                instsum.append(ii)
            if isinstance(jj, tuple):
                for j in jj:
                    instsum.append(j)
            elif isinstance(jj, int):
                instsum.append(jj)
            sum.append(tuple(instsum))
        return tuple(sum)

    def create_name_from_pair(self, arg1, arg2):
        sum = []
        for arg in (arg1, arg2):
            if isinstance(arg, tuple):
                for i in arg:
                    sum.append(i)
            elif isinstance(arg, str):
                sum.append(arg)
        return tuple(sum)

    def gen_pairing_names(self, pairings, node_list):
        pairing_names = []
        pairing_count = 0
        for i, j in pairings:
            pairing_count += 1
            if i is not None and j is not None:
                try:
                    pairing_names.append((node_list[i], node_list[j]))
                except IndexError:
                    pass
            elif i is None and j is not None:
                try:
                    pairing_names.append((None, node_list[j]))
                except IndexError:
                    pass
            elif j is None and i is not None:
                try:
                    pairing_names.append((node_list[i], None))
                except IndexError:
                    pass
            else:
                pairing_names.append((None, None))
        return pairing_names, pairing_count

    def run_pharma_tree(self):
        """Take all active sites and join them randomly. This is a breadth
        first reverse-tree algorithm."""
        done = False
        t1 = time()
        tree = Tree()
        node_list = self._active_sites.keys()
        pairings = tree.branchify(node_list) # initial pairing up of active sites
        pairing_names, pairing_count = self.gen_pairing_names(pairings, node_list)
        pass_count = 0  # count the number of times the loop joins all bad pairs of active sites
        pharma_sites = {key:range(len(val)) for key, val in self._active_sites.items()}
        while not done:
            # loop over pairings, each successive pairing should narrow down the active sites
            no_pairs = self.combine_pairs(pairing_names, pharma_sites)
            # count the number of times no pairings were made
            pairings = tree.branchify(node_list)
            if no_pairs == pairing_count:
                pass_count += 1
            # TESTME(pboyd): This may really slow down the routine.
            else:
                # re-set if some good sites were found
                pass_count = 0
            # TESTME(pboyd)
            if pass_count == self.max_pass_count or pairings is None:
                done = True
            else:
                node_list = pharma_sites.keys()
                
                pairing_names, pairing_count = self.gen_pairing_names(pairings, 
                                                                      node_list)

        t2 = time()
        self.time = t2 - t1
        return pharma_sites.values(), pharma_sites.keys()
        #return self._active_sites.values(), self._active_sites.keys() 

    def obtain_pharma_error(self, name, sites):
        if not isinstance(name, tuple):
            return 0.
        base = self._active_sites[name[0]] % [j[0] for j in sites]
        base.shift_by_centre_of_atoms()
        mean_errors = []
        for ii in range(1, len(name)):
            atms = [q[ii] for q in sites]
            match = self._active_sites[name[ii]] % atms
            match.shift_by_centre_of_atoms()
            R = rotation_from_vectors(match._coordinates[:], 
                                      base._coordinates[:])
            match.rotate(R)
            mean_errors.append(np.mean(base._coordinates - match._coordinates))
        return np.mean(mean_errors)

    def obtain_co2_distribution(self, name, sites):

        grid = np.zeros((10, 10, 10))
        if not isinstance(name, tuple):
            return grid
        base = self._active_sites[name[0]] % [j[0] for j in sites]
        base.shift_by_centre_of_atoms()
        mean_errors = []
        for ii in range(1, len(name)):
            atms = [q[ii] for q in sites]
            match = self._active_sites[name[ii]] % atms
            match.shift_by_centre_of_atoms()
            T = match.centre_of_atoms
            R = rotation_from_vectors(match._coordinates[:], 
                                      base._coordinates[:])
            match.rotate(R)
            co2 = self._co2_sites[name[ii]]
            co2 -= T
            co2 = np.dot(R[:3,:3], co2.T).T
            # bin the x, y, and z
            print co2
        
        return grid 

    def write_final_files(self, names, pharma_graphs, site_count): 
        pickle_dic = {}
        co2_dist_dic = {}
        filebasename='%s_N.%i_r.%3.2f_ac.%i_ma.%i_rs.%i_t.%3.2f'%("pharma",
                                                 site_count,
                                                 self.radii,
                                                 self.min_atom_cutoff,
                                                 self.max_pass_count,
                                                 self.seed,
                                                 self.tol)
        f = open(filebasename+".csv", 'w')
        f.writelines("length,pharma_std,el_avg,el_std,vdw_avg,vdw_std,tot_avg,tot_std,name\n") 
        for id, (name, pharma) in enumerate(zip(names, pharma_graphs)):
            nn = name[0] if isinstance(name, tuple) else name
            jj = [i[0] if isinstance(i, tuple) else i for i in pharma]
            
            site = self._active_sites[nn] % jj
            site.shift_by_centre_of_atoms()
            pickle_dic[name] = site 
            el, vdw, tot = [], [], []
            if isinstance(name, tuple):
                [el.append(self.el_energy[i]) for i in name]
                [vdw.append(self.vdw_energy[i]) for i in name]
                [tot.append(self.vdw_energy[i]+self.el_energy[i]) for i in name]

                elavg = np.mean(el)
                elstd = np.std(el)
                vdwavg = np.mean(vdw)
                vdwstd = np.std(vdw)
                totavg = np.mean(tot)
                totstd = np.std(tot)
            else:
                elavg = self.el_energy[name] 
                elstd = 0. 
                vdwavg = self.vdw_energy[name]
                vdwstd = 0. 
                totavg = self.vdw_energy[name] + self.el_energy[name]
                totstd = 0.
            error = self.obtain_pharma_error(name, pharma)
            co2_dist = self.obtain_co2_distribution(name, pharma)
            co2_dist_dic[name] = co2_dist
            f.writelines("%i,%f,%f,%f,%f,%f,%f,%f,%s\n"%(len(name),error,elavg,elstd,vdwavg,
                                                         vdwstd,totavg,totstd,name))

        f.close()
        f = open(filebasename+".pkl", 'wb')
        pickle.dump(pickle_dic, f)
        f.close()
        f = open(filebasename+"_co2dist.pkl", 'wb')
        pickle.dump(co2_dist_dic, f)
        f.close()
        # write an energy distribution file


#class MPIPharmacophore(Pharmacophore, MPITools):
#    """Each rank has it's own set of active sites, these need to be tracked for 
#    pairings across nodes, which will either happen at each new pairing iteration,
#    or once the set of all binding sites on a node have been reduced fully."""
#
#    def generate_node_list(self):
#        return_list = []
#        for name in self._active_sites.keys():
#            return_list.append((name, rank))
#        return return_list
#
#    def data_distribution(self, pairing_names):
#        """Organize the data distribution for this loop"""
#        chunks = []
#        # format (from_node, active_site_name, to_node)
#        node_transmissions = [] 
#        sz = int(math.ceil(float(len(list(pairing_names)))/ float(size)))
#        if sz == 0:
#            sz = 1
#        # optimize where to send each chunks.?
#        for nt, i in enumerate(self.chunks(pairing_names, sz)):
#            chunks.append(i)
#            for n1, n2 in i:
#                try:
#                    name, nf = n1
#                    if nf == nt:
#                        pass
#                    else:
#                        node_transmissions.append((nf, name, nt))
#
#                except TypeError:
#                    pass
#                try:
#                    name, nf = n2
#                    if nf == nt:
#                        pass
#                    else:
#                        node_transmissions.append((nf, name, nt))
#                except TypeError:
#                    pass
#        # shitty hack for making the last node do nothing
#        for i in range(size-len(chunks)):
#            chunks.append([(None,None)])
#
#        return chunks, node_transmissions 
#        # need to tell the nodes where to send and recieve their
#        # active sites.
#
#    def assign_unique_ids(self, node_list):
#        return {name:id for id, (name, p) in enumerate(node_list)} 
#
#    def rank_zero_stuff(self, node_list):
#        tree = Tree()
#        node_list = [j for i in node_list for j in i]
#        uuids = self.assign_unique_ids(node_list)
#        pairings = tree.branchify(node_list) # initial pairing up of active sites
#        pairing_names, pairing_count = self.gen_pairing_names(pairings, node_list)
#        return pairing_names, pairing_count, uuids
#
#    def run_pharma_tree(self):
#        """Take all active sites and join them randomly. This is a breadth
#        first reverse-tree algorithm."""
#        # rank 0 will keep a list of all the current active sites available, partition them, 
#        # and direct which nodes to pair what active sites. The other nodes will then 
#        # swap active sites, and pair. The resulting list of pairs will be sent back to
#        # rank 0 for future pairings.
#        t1 = time()
#        done = False
#        to_root = self.generate_node_list()
#        # collect list of nodes and all energies to the mother node. 
#        node_list = comm.gather(to_root, root=0)
#        energies = comm.gather(self.energy, root=0)
#        # perform the pairing centrally, then centrally assign pairings to different nodes
#        # maybe a smart way is to minimize the number of mpi send/recv calls by sending
#        # pairs off to nodes which possess one or more of the sites already.
#        if rank == 0:
#            [self.energy.update(i) for i in energies]
#            pairing_names, pairing_count, uuids = self.rank_zero_stuff(node_list)
#            chunks, node_transmissions = self.data_distribution(pairing_names)
#
#        while not done:
#            # loop over pairings, each successive pairing should narrow down the active sites
#            # broadcast the pairing to the nodes.
#            if rank != 0:
#                uuids, chunks, node_transmissions = None, None, None
#            # broadcast the uuids
#            uuids = comm.bcast(uuids, root=0)
#            # Some MPI stuff
#            pairings = comm.scatter(chunks, root=0)
#            node_transmissions = comm.bcast(node_transmissions, root=0)
#            self.collect_recieve(node_transmissions, uuids)
#            no_pairs = self.combine_pairs(pairings)
#           
#            no_pairs = comm.gather(no_pairs, root=0)
#            to_root = self.generate_node_list()
#            node_list = comm.gather(to_root, root=0)
#            if rank == 0:
#                no_pairs = sum(no_pairs)
#                # for some reason the number of no_pairs can get larger than
#                # the number of pairings broadcast to each node...?
#                if no_pairs >= pairing_count:
#                    pass_count += 1
#                # TESTME(pboyd): This may really slow down the routine.
#                else:
#                # re-set if some good sites were found
#                    pass_count = 0
#                if pass_count == self.max_pass_count:
#                    done = True
#                else:
#                    pairing_names, pairing_count, uuids = self.rank_zero_stuff(node_list)
#                    chunks, node_transmissions = self.data_distribution(pairing_names)
#            # broadcast the complete list of nodes and names to the other nodes.
#            done = comm.bcast(done, root=0)
#        t2 = time()
#        self.time = t2 - t1
#        # collect all the nodes and write some fancy stuff.
#        sites = comm.gather(self._active_sites, root=0)
#        if rank == 0:
#            for i in sites:
#                self._active_sites.update(i)
#            return self._active_sites.values(), self._active_sites.keys() 
#        return None, None
#        
#    def collect_recieve(self, node_transmissions, uuids):
#        """Send active sites on this node that are not in pairings, collect
#        active sites which are in pairings but not on this node."""
#        for n_from, name, n_to in node_transmissions:
#            tag_id = uuids[name]
#            if rank == n_from:
#                sending = self._active_sites.pop(name)
#                comm.send({name:sending}, dest=n_to, tag=tag_id)
#            elif rank == n_to:
#                site = comm.recv(source=n_from, tag=tag_id)
#                self._active_sites.update(site)
#
#
#    def combine_pairs(self, pairings):
#        """Combining pairs of active sites"""
#        no_pairs = 0
#        for (i, j) in pairings:
#            if i is not None and j is not None:
#                g1, g2 = self._active_sites[i[0]], self._active_sites[j[0]]
#                p = self.get_clique(g1, g2)
#                # check if the clique is greater than the number of
#                # atoms according to min_cutoff
#                if len(p) >= self.min_atom_cutoff:
#                    #(g1 % p).debug("Pharma")
#                    newnode = ( g1 % p )
#                    newname = self.create_name_from_pair(i[0], j[0])
#                    del self._active_sites[j[0]]
#                    del self._active_sites[i[0]]
#                    self._active_sites.update({newname:newnode})
#                else:
#                    no_pairs += 1
#            elif i is None and j is None:
#                continue
#            elif i is None:
#                no_pairs += 1
#            elif j is None:
#                no_pairs += 1
#
#        #print "rank %i"%rank, "pairings %i"%(len(pairings)), "no_pairs count %i"%no_pairs
#        return no_pairs

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
    directory.dir_scan()
    for mof, path in directory.mof_dirs:
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
    site_count = 0
    if binding_sites.absl_calc_exists():
        binding_sites.co2_xyz_parser()
        binding_sites.energy_parser()
        for site in binding_sites.sites:
            try:
                el = site.pop('electrostatic')
                vdw = site.pop('vanderwaals')
                if el + vdw < -30.:
                    coords = np.array([site[i] for i in ['C', 'O1', 'O2']])
                    active_site = pharma.get_active_site(coords, faps_graph)
                    # set all elements to C so that the graph matching is non-discriminatory
                    #active_site.set_elem_to_C()
                    pharma.store_active_site(active_site, 
                                             name="%s.%i"%(mof.name, site_count),
                                             el_energy=el,
                                             vdw_energy=vdw)
                    pharma.store_co2_pos(coords,
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
    for mof, path in directory.mof_dirs:
        faps_mof = Structure(name=mof)
        faps_mof.from_cif(os.path.join(path, mof+".cif"))
        add_to_pharmacophore(faps_mof, pharma, path)
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
    if rank == 0:
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
    #main()

