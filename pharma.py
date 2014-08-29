#!/usr/bin/env python

import os
import shutil
import sys
import pickle
import math
from optparse import OptionParser
from post_processing import Fastmc_run
import itertools
import networkx as nx
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle
from scipy.spatial import distance
import subprocess
from math import pi
from nets import Net
from sub_graphs import SubGraph
#from correspondence_graphs import CorrGraph
sys.path.append("/home/pboyd/codes_in_development/mcqd_api/build/lib.linux-x86_64-2.7")
import _mcqd as mcqd

from faps import Structure, Atom, Cell, mk_gcmc_control, Guest
from config_fap import Options

#SQL stuff
from sql_backend import Data_Storage, SQL_Pharma, SQL_ActiveSite, SQL_ActiveSiteAtoms, SQL_Distances, SQL_ActiveSiteCO2
#MPI stuff
from mpi import MPIPharmacophore, MPIMOFDiscovery

global MPIrank, MPIsize, comm
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

class Pharmacophore(object):

    """Grab atoms surrounding binding sites. compare with maximal clique detection."""
    def __init__(self, radii=4.5, min_atom_cutoff=9, max_pass_count=20, random_seed=None, tol=0.4):
        self.site_count = 0
        self.el_energy = {}
        self.vdw_energy = {}
        self._active_sites = {} # stores all subgraphs in a dictionary of lists.  The keys are the mof names
        self.pharma_sites = {} # key: active site name (mofname.[int]) val: range of graph size
        self._co2_sites = {}
        self._radii = radii # radii to scan for active site atoms
        self._bad_pair = {} # store all binding site pairs which did not result in a pharmacophore
        self.max_pass_count = max_pass_count 
        self.min_atom_cutoff = min_atom_cutoff
        self.lattices = {}  # store the lattice information for energetic calculations at the end of the run.
        self.time = 0.
        self.tol = tol
        self.grid = (10, 10, 10)
        self.node_done = False # for MPI implementation
        # initialize a random seed (for testing purposes)
        if random_seed is None:
            # make the seed time-based, but still be able to recover what it
            # is for tracking purposes
            self.seed = random.randint(0, sys.maxint)
        else:
            self.seed = random_seed
        # set the random seed
        random.seed(self.seed)
        self.sql_active_sites = Data_Storage("active_sites") 

    @property
    def radii(self):
        return self._radii
    @radii.setter
    def radii(self, val):
        self._radii = val

    def get_main_graph(self, faps_obj):
        """Takes a structure object and converts to a sub_graph using
        routines borrowed from net_finder."""
        self.lattices[faps_obj.name] = faps_obj.cell.cell
        s = SubGraph(name=faps_obj.name)
        # in the future, determine the size of the supercell which would fit the radii property
        s.from_faps(faps_obj, supercell=(3,3,3))
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
        #self.el_energy[name] = el_energy
        #self.vdw_energy[name] = vdw_energy
        #self._active_sites[name] = range(len(activesite)) # activesite
        self.pharma_sites[name] = range(len(activesite))

        s = SQL_ActiveSite(name, len(activesite), vdw_energy, el_energy)
        self.sql_active_sites.store(s)
        for id, element in enumerate(activesite.elements):
            charge = activesite.charges[id]
            pos = activesite._coordinates[id]
            s = SQL_ActiveSiteAtoms(pos, element, charge, id, name)
            self.sql_active_sites.store(s)

        for (x, y), dist in np.ndenumerate(activesite._dmatrix):
            s = SQL_Distances(x, y, dist, name)
            self.sql_active_sites.store(s)
    
    def store_co2_pos(self, pos, name='Default'):
        #self._co2_sites[name] = pos
        #print "Name: %s"%(name)
        #print "pos = C (%12.5f, %12.5f, %12.5f)"%(tuple(pos[0]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(pos[1]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(pos[2]))

        s = SQL_ActiveSiteCO2(name, pos)
        self.sql_active_sites.store(s)

    def get_clique(self, g1, g2):
        """Returns the atoms in g1 which constitute a maximal clique
        with respect to g2 and a given tolerance.

        """
        nodes = mcqd.correspondence(g1.elements, g2.elements)
        if not nodes:
            return [], []
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
        # catch-all?
        return [], []

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

    def get_active_site_from_sql(self, name):
        sqlas = self.sql_active_sites.get_active_site(name)
        # convert to subgraph
        #graph = SubGraph(name=sqlas.name)
        #graph.elements = range(sqlas.size)
        #graph.charges = range(sqlas.size)
        #graph._orig_index = range(sqlas.size)
        #graph._new_index = range(sqlas.size)
        #graph._coordinates = np.empty((sqlas.size, 3))
        #graph._dmatrix = np.empty((sqlas.size, sqlas.size))
        #for atom in sqlas.atoms:
        #    coord = np.array([atom.x, atom.y, atom.z])
        #    graph._coordinates[atom.orig_id] = coord
        #    graph.elements[atom.orig_id] = atom.elem
        #    graph.charges[atom.orig_id] = atom.charge
        #for dist in sqlas.distances:
        #    graph._dmatrix[dist.row,dist.col] = dist.dist

        co2 = np.empty((3,3))
        sqlco2 = sqlas.co2[0]
        co2[0][0] = sqlco2.cx
        co2[0][1] = sqlco2.cy
        co2[0][2] = sqlco2.cz
        co2[1][0] = sqlco2.o1x
        co2[1][1] = sqlco2.o1y
        co2[1][2] = sqlco2.o1z
        co2[2][0] = sqlco2.o2x
        co2[2][1] = sqlco2.o2y
        co2[2][2] = sqlco2.o2z
        #print "Name: %s"%(name)
        #print "pos = C (%12.5f, %12.5f, %12.5f)"%(tuple(co2[0]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(co2[1]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(co2[2]))
        return co2, sqlas.vdweng, sqlas.eleng
    
    def get_active_site_graph_from_sql(self, name):
        sqlas = self.sql_active_sites.get_active_site(name)
        # convert to subgraph
        graph = SubGraph(name=sqlas.name)
        graph.elements = range(sqlas.size)
        graph.charges = range(sqlas.size)
        graph._orig_index = range(sqlas.size)
        graph._new_index = range(sqlas.size)
        graph._coordinates = np.empty((sqlas.size, 3))
        graph._dmatrix = np.empty((sqlas.size, sqlas.size))
        for atom in sqlas.atoms:
            coord = np.array([atom.x, atom.y, atom.z])
            graph._coordinates[atom.orig_id] = coord
            graph.elements[atom.orig_id] = atom.elem
            graph.charges[atom.orig_id] = atom.charge
        for dist in sqlas.distances:
            graph._dmatrix[dist.row,dist.col] = dist.dist
        return graph

    def combine_pairs(self, pairings, sites):
        """Combining pairs of active sites"""
        no_pairs = 0
        for (i, j) in pairings:
            if i is not None and j is not None:
                # get the representative names, representative atoms
                namei, namej = self.get_rep_site_name(i), self.get_rep_site_name(j)
                # get these from SQL DB.

                #asi, qq, qq, qq = self.get_active_site_from_sql(namei)
                #asj, qq, qq, qq = self.get_active_site_from_sql(namej)
                try:
                    asi = self._active_sites[namei]
                except KeyError:
                    asi = self.get_active_site_graph_from_sql(namei)
                    self._active_sites[namei] = asi
                try:
                    asj = self._active_sites[namej]
                except KeyError:
                    asj = self.get_active_site_graph_from_sql(namej)
                    self._active_sites[namej] = asj

                nodesi, nodesj = self.get_rep_nodes(i, sites), self.get_rep_nodes(j, sites)
                g1, g2 = (asi % nodesi), (asj % nodesj)
                p,q = self.get_clique(g1, g2)
                # check if the clique is greater than the number of
                # atoms according to min_cutoff
                if len(p) >= self.min_atom_cutoff:
                    newnode = self.create_node_from_pair([sites[i][xx] for xx in p], 
                                                         [sites[j][yy] for yy in q])
                    newname = self.create_name_from_pair(i, j)
                    sites.update({newname:newnode})
                    del sites[i]
                    del sites[j]
                    # note - the new node will have site [namei] as the representative
                    # so we delete site [namej] for memory purposes
                    del self._active_sites[namej]

                # append to the bad_pair dictionary otherwise
                else:
                    no_pairs += 1
            # in cases where the number of active sites is odd,
            # the itertools function will produce a pair that has
            # (int, None). The following is to account for that.
            # mpi routines will sometimes send (None, None)'s if 
            # the number of nodes is greater than the number of pairs
            # to compute.. 
            elif i is None and j is None:
                continue
            elif i is None:
                no_pairs += 1
            elif j is None:
                no_pairs += 1
        return no_pairs, sites

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
                pairing_names.append((node_list[i], node_list[j]))
            elif i is None and j is not None:
                pairing_names.append((None, node_list[j]))
            elif j is None and i is not None:
                pairing_names.append((node_list[i], None))
            else:
                pairing_names.append((None, None))
        return pairing_names, pairing_count

    def run_pharma_tree(self):
        """Take all active sites and join them randomly. This is a breadth
        first reverse-tree algorithm."""
        done = False
        t1 = time()
        tree = Tree()
        #pharma_sites = {key:range(len(val)) for key, val in self._active_sites.items()}
        pharma_sites = self.pharma_sites
        node_list = sorted(pharma_sites.keys())
        pairings = tree.branchify(node_list) # initial pairing up of active sites
        pairing_names, pairing_count = self.gen_pairing_names(pairings, node_list)
        pass_count = 0  # count the number of times the loop joins all bad pairs of active sites
        flush_count = 0
        while not done:
            flush_count += 1
            # loop over pairings, each successive pairing should narrow down the active sites
            no_pairs, pharma_sites = self.combine_pairs(pairing_names, pharma_sites)
            if (flush_count % 1000 == 0):
                self.sql_active_sites.flush()
            # count the number of times no pairings were made
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
                node_list = sorted(pharma_sites.keys())
                pairings = tree.branchify(node_list)
                
                pairing_names, pairing_count = self.gen_pairing_names(pairings, 
                                                                      node_list)

        t2 = time()
        self.time = t2 - t1
        return pharma_sites.values(), pharma_sites.keys()
        #return self._active_sites.values(), self._active_sites.keys() 

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
def site_xyz(dic):
    f = open("binding_site.xyz", "w")
    f.writelines("%i\n"%(len(dic.keys())))
    f.writelines("binding site\n")
    for key, val in dic.items():
        f.writelines("%s %9.4f %9.4f %9.4f\n"%(key[:1], val[0], val[1], val[2]))
    f.close()

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
    #main()

