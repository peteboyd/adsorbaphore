#!/usr/bin/env python

import os
import shutil
import sys
import pickle
import itertools
import operator
from time import time
import numpy as np
import random
from random import shuffle
from scipy.spatial import distance
from sub_graphs import SubGraph
#from correspondence_graphs import CorrGraph
sys.path.append("/home/pboyd/codes_in_development/mcqd_api/build/lib.linux-x86_64-2.7")
import _mcqd as mcqd

#SQL stuff
from sql_backend import Data_Storage, SQL_Pharma, SQL_ActiveSite, SQL_ActiveSiteAtoms, SQL_Distances, SQL_ActiveSiteCO2
from sql_backend import SQL_Adsorbophore, SQL_AdsorbophoreSite, SQL_AdsorbophoreSiteIndices
#Data analysis stuff
from post_processing import Fastmc_run, PairDistFn

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
        self._distance_matrix = {}
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
        filebasename='%s_r.%3.2f'%('active_sites',self.radii)
        self.sql_active_sites = Data_Storage(filebasename) 

    @property
    def radii(self):
        return self._radii
    @radii.setter
    def radii(self, val):
        self._radii = val

    def compute_supercell(self, mof, supercell):
        """Returns a list of original indices from the mof as well as
        coordinates of the supercell in a numpy array.

        """
        cell = mof.cell.cell
        inv_cell = np.linalg.inv(cell.T)
        multiplier = reduce(operator.mul, supercell, 1)
        size = len(mof.atoms)
        original_indices = np.empty(size*multiplier, dtype=np.int)
        coordinates = np.empty((size*multiplier, 3), dtype=np.float)
        
        # shift by the median supercell - this is so that the binding sites
        # are covered on all sides
        box_shift = np.rint(np.median(np.array([(0,0,0), [k-1 for k in supercell]]), axis=0))
        super_box = list(itertools.product(*[itertools.product(range(j)) 
                           for j in supercell]))
        for id, atom in enumerate(mof.atoms):
            for mult, box in enumerate(super_box):
                fpos = atom.ifpos(inv_cell) + np.array([b[0] for b in box], dtype=np.float) - box_shift
                original_indices[id + mult*size] = id
                coordinates[id + mult*size] = np.dot(fpos, cell)
        return original_indices, coordinates

    def get_main_graph(self, faps_obj):
        """Takes a structure object and converts to a sub_graph using
        routines borrowed from net_finder."""
        self.lattices[faps_obj.name] = faps_obj.cell.cell
        #s = SubGraph(name=faps_obj.name)
        # in the future, determine the size of the supercell which would fit the radii property
        #s.from_faps(faps_obj, supercell=(3,3,3))
        return s
    
    def get_active_site(self, binding_site, mof_coordinates):
        """Takes all atoms within a radii of the atoms in the binding site."""
        distances = distance.cdist(binding_site, mof_coordinates)
        sub_idx = []
        coords = []
        for (x, y), val in np.ndenumerate(distances):
            if val <= self.radii:
                sub_idx.append(y)
                coords.append(mof_coordinates[y])
        sub_idx = list(set(sub_idx))
        return sub_idx, coords

    def store_active_site(self, activesite, dmatrix, name='Default', mof_path="./", el_energy=0., vdw_energy=0.):
        """(ind, x, y, z, element, [mofind], charge)"""
        self.site_count += 1
        #self.el_energy[name] = el_energy
        #self.vdw_energy[name] = vdw_energy
        #self._active_sites[name] = range(len(activesite)) # activesite
        self._active_sites[name] = activesite
        self.pharma_sites[name] = range(len(activesite))
        self._distance_matrix[name] = dmatrix

        s = SQL_ActiveSite(name, len(activesite), mof_path, vdw_energy, el_energy)
        self.sql_active_sites.store(s)
        for (i, x, y, z, element, mofind, charge) in (activesite):
            s = SQL_ActiveSiteAtoms((x, y, z), element, charge, i, mofind, name)
            self.sql_active_sites.store(s)

        #for (x, y), dist in np.ndenumerate(dmatrix):
        #    s = SQL_Distances(x, y, dist, name)
        #    self.sql_active_sites.store(s)
    
    def store_co2_pos(self, pos, name='Default'):
        #self._co2_sites[name] = pos
        #print "Name: %s"%(name)
        #print "pos = C (%12.5f, %12.5f, %12.5f)"%(tuple(pos[0]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(pos[1]))
        #print "pos = O (%12.5f, %12.5f, %12.5f)"%(tuple(pos[2]))

        s = SQL_ActiveSiteCO2(name, pos)
        self.sql_active_sites.store(s)

    def get_clique(self, asi, nodesi, disti, asj, nodesj, distj):
        """Returns the atoms in g1 which constitute a maximal clique
        with respect to g2 and a given tolerance.
        activesite = (ind, x, y, z, element, [mofind], charge)
        """
        g1_elem = [asi[i][4] for i in nodesi]
        g2_elem = [asj[i][4] for i in nodesj]
        nodes = mcqd.correspondence(g1_elem, g2_elem)
        if not nodes:
            return [], []

        g1_dist = disti[nodesi,:][:,nodesi]
        g2_dist = distj[nodesj,:][:,nodesj]
        adj_matrix = mcqd.correspondence_edges(nodes,
                                               g1_dist, 
                                               g2_dist,
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
        return co2, sqlas.vdweng, sqlas.eleng
    
    def get_active_site_graph_from_sql(self, name):
        """(ind, x, y, z, element, [mofind], charge)"""
        sqlas = self.sql_active_sites.get_active_site(name)
        # convert to subgraph
        #graph = SubGraph(name=sqlas.name)
        #graph.elements = range(sqlas.size)
        #graph.charges = range(sqlas.size)
        #graph._orig_index = range(sqlas.size)
        #graph._new_index = range(sqlas.size)
        #graph._coordinates = np.empty((sqlas.size, 3))
        graph = []
        dmatrix = np.empty((sqlas.size, sqlas.size))
        for atom in sqlas.atoms:
            site = (atom.id, atom.x, atom.y, atom.z, atom.elem, atom.mof_id, atom.charge)
            graph.append(site)
        for dist in sqlas.distances:
            dmatrix[dist.row,dist.col] = dist.dist
        return sorted(graph), dmatrix

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
                    disti = self._distance_matrix[namei]
                except KeyError:
                    asi, disti = self.get_active_site_graph_from_sql(namei)
                    self._active_sites[namei] = asi
                    self._distance_matrix[namei] = disti
                try:
                    asj = self._active_sites[namej]
                    distj = self._distance_matrix[namej]
                except KeyError:
                    asj, distj = self.get_active_site_graph_from_sql(namej)
                    self._active_sites[namej] = asj
                    self._distance_matrix[namej] = distj
                nodesi, nodesj = self.get_rep_nodes(i, sites), self.get_rep_nodes(j, sites)
                p,q = self.get_clique(asi, nodesi, disti, asj, nodesj, distj)
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
                    del self._distance_matrix[namej]
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
        return pharma_sites
        #return self._active_sites.values(), self._active_sites.keys() 

    def store_adsorbophores(self, adsorbophores):
        """Store the raw form of the adsorbophores - which active_sites are contained,
        which indices of the active sites are included.
        
        [rank] --> pharma_sites --> indices
        """
        filebasename='%s_N.%i_r.%3.2f_ac.%i_ma.%i_rs.%i_t.%3.2f'%('adsorbophores',
                                                                  self.site_count,
                                                                  self.radii,
                                                                  self.min_atom_cutoff,
                                                                  self.max_pass_count,
                                                                  self.seed,
                                                                  self.tol)
        data_storage = Data_Storage(filebasename)
        names = [i for i in adsorbophores.keys() if isinstance(i, tuple)]
        names = sorted(names, key=len)
        # rank by size of tuple.
        names.reverse()
        for rank, name in enumerate(names):
            vals = adsorbophores[name]
            ads = SQL_Adsorbophore(rank)
            data_storage.store(ads)
            for id, nn in enumerate(name):
                peanut = SQL_AdsorbophoreSite(rank, nn)
                data_storage.store(peanut)
                for order, site_index in enumerate([jj[id] for jj in vals]):
                    butter = SQL_AdsorbophoreSiteIndices(nn, order, site_index)
                    data_storage.store(butter)
            data_storage.flush()


