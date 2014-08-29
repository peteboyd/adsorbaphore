from pharma import Pharmacophore
import os
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPIrank = comm.Get_rank()
    MPIsize = comm.Get_size()
except ImportError:
    print "Warning! no module named mpi4py found, I hope you are not running this in parallel!"
    pass

class MPITools(object):
    
    def chunks(self, l, n):
        """yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i+n]
    
class MPIMOFDiscovery(MOFDiscovery, MPITools):
    
    def dir_scan(self, max_mofs=None):
        chunks, ranks = None, None
        if MPIrank == 0:
            f = open("MOFs_used.csv", "w")
            count = 0 
            for root, directories, filenames in os.walk(self.directory):
                mofdir = os.path.basename(root)
                if self._valid_mofdir(mofdir):
                    f.writelines("%s\n"%mofdir)
                    self.mof_dirs.append((mofdir,root))
                    count += 1
                if (max_mofs is not None) and (count == max_mofs):
                    break
            f.close()
            sz = int(math.ceil(float(len(self.mof_dirs)) / float(MPIsize)))
            ranks, chunks = [], []
            for rr, ch in enumerate(self.chunks(self.mof_dirs, sz)):
                ranks += [rr for i in range(len(ch))]
                chunks.append(ch)
            for kk in range(MPIsize - len(chunks)):
                chunks.append(tuple([(None, [])]))
        self.mof_dirs = comm.scatter(chunks, root=0)

class MPIPharmacophore(Pharmacophore, MPITools):
    """Each rank has it's own set of active sites, these need to be tracked for 
    pairings across nodes, which will either happen at each new pairing iteration,
    or once the set of all binding sites on a node have been reduced fully."""

    def generate_node_list(self):
        return_list = []
        for name in self._active_sites.keys():
            return_list.append((name, MPIrank))
        return return_list

    def data_distribution(self, pairing_names, node_list):
        """Organize the data distribution for this loop"""
        chunks = []
        # format (from_node, active_site_name, to_node)
        # for some reason, the function 'chunks' produces duplicate
        # pairing names, which are then distributed over different nodes
        # this creates a duplication problem, as well as the difficulty that
        # _active sites are deleted from the node after they are sent across
        # hence the dictionary to remove any redundancy.
        node_transmissions = {}
        keep = {}
        sz = int(math.ceil(float(len(list(pairing_names)))/ float(MPIsize)))
        if sz == 0:
            sz = 1
        # optimize where to send each chunks.?
        dupes = {}
        for nt, i in enumerate(self.chunks(pairing_names, sz)):
            chunks.append(i)
            for n1, n2 in i:
                if isinstance(n1, tuple):
                    name1 = n1[0]
                else:
                    name1 = n1
                if isinstance(n2, tuple):
                    name2 = n2[0]
                else:
                    name2 = n2
                if name1 is not None:
                    dupes.setdefault(name1, 0)
                    dupes[name1] += 1
                    nf = [i[1] for i in node_list if i[0] == name1][0]
                    if nf != nt:
                        node_transmissions[name1] = (nf, name1, nt)
                    else:
                        keep[name1] = (nf, name1, nt)
                if name2 is not None:
                    dupes.setdefault(name2, 0)
                    dupes[name2] += 1
                    nf = [i[1] for i in node_list if i[0] == name2][0]
                    if nf != nt:
                        node_transmissions[name2] = (nf, name2, nt)
                    else:
                        keep[name2] = (nf, name2, nt)
        # shitty hack for making the last node do nothing
        for i in range(MPIsize-len(chunks)):
            chunks.append([(None,None)])
        return chunks, node_transmissions.values() 
        # need to tell the nodes where to send and recieve their
        # active sites.

    def assign_unique_ids(self, node_list):
        return {name:((i+1)*(p+1)) for i, (name, p) in enumerate(node_list)} 

    def rank_zero_stuff(self, tree, pharma_sites, node_list):
        if node_list:
            # the list is sorted so that the results of a parallel run will coincide with that of 
            # a serial run using the same random seed.
            node_list = [j for i in node_list for j in i]
            psort = sorted(pharma_sites.keys())
            uuids = self.assign_unique_ids(node_list)
            pairings = tree.branchify(psort)
            pairing_names, pairing_count = self.gen_pairing_names(pairings, psort)
        return pairing_names, pairing_count, uuids, node_list

    def local_tree(self, tree):
        """This is just the serial version of the tree"""
        done = False
        t1 = time()
        tree = Tree()
        pharma_sites = {key:range(len(val)) for key, val in self._active_sites.items()}
        node_list = sorted(pharma_sites.keys())
        pairings = tree.branchify(node_list) # initial pairing up of active sites
        pairing_names, pairing_count = self.gen_pairing_names(pairings, node_list)
        pass_count = 0  # count the number of times the loop joins all bad pairs of active sites
        while not done:
            # loop over pairings, each successive pairing should narrow down the active sites
            no_pairs, pharma_sites = self.combine_pairs(pairing_names, pharma_sites)
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
        self.node_done = True
        return pharma_sites 

    def run_pharma_tree(self):
        """Take all active sites and join them randomly. This is a breadth
        first reverse-tree algorithm."""
        # rank 0 will keep a list of all the current active sites available, partition them, 
        # and direct which nodes to pair what active sites. The other nodes will then 
        # swap active sites, and pair. The resulting list of pairs will be sent back to
        # rank 0 for future pairings.
        tree = Tree()
        t1 = time()
        pharma_sites = {key:range(len(val)) for key, val in self._active_sites.items()}
        pharma_sites = self.collect_broadcast_dictionary(pharma_sites)
        to_root = self.generate_node_list()
        # collect list of nodes and all energies to the mother node. 
        node_list = comm.gather(to_root, root=0)
        # perform the pairing centrally, then centrally assign pairings to different nodes
        # maybe a smart way is to minimize the number of mpi send/recv calls by sending
        # pairs off to nodes which possess one or more of the sites already.
        if MPIrank == 0:
            pairing_names, pairing_count, uuids, node_list = self.rank_zero_stuff(tree, pharma_sites, node_list)
            chunks, node_transmissions = self.data_distribution(pairing_names, node_list)

        done = False
        while not done:
            # loop over pairings, each successive pairing should narrow down the active sites
            # broadcast the pairing to the nodes.
            if MPIrank != 0:
                uuids, chunks, node_transmissions = None, None, None
            # broadcast the uuids
            uuids = comm.bcast(uuids, root=0)
            # Some MPI stuff
            pairings = comm.scatter(chunks, root=0)
            node_transmissions = comm.bcast(node_transmissions, root=0)
            self.collect_recieve(node_transmissions, uuids)
            to_root = self.generate_node_list()
            node_list = comm.gather(to_root, root=0)
            # have to determine which sites are not being paired in this node, 
            # then delete these sites before sending the list back to node 0
            pairing_nodes = []
            for k, l in pairings:
                pairing_nodes.append(k)
                pairing_nodes.append(l)
            
            # Do a local reduction of sites until no reductions can be done.. then
            # send it to a global reduction
            if not self.node_done:
                pharma_sites = self.local_tree(tree)
                no_pairs = 0
            else:
                # actual clique finding
                pop_nodes = [ps for ps in pharma_sites.keys() if ps not in pairing_nodes] 
                no_pairs, pharma_sites = self.combine_pairs(pairings, pharma_sites)
                [pharma_sites.pop(ps) for ps in pop_nodes]
            pharma_sites = self.collect_broadcast_dictionary(pharma_sites)
            no_pairs = comm.gather(no_pairs, root=0)
            if MPIrank == 0:
                no_pairs = sum(no_pairs)
                # for some reason the number of no_pairs can get larger than
                # the number of pairings broadcast to each node...?
                if no_pairs >= pairing_count:
                    pass_count += 1
                # TESTME(pboyd): This may really slow down the routine.
                else:
                # re-set if some good sites were found
                    pass_count = 0
                if pass_count == self.max_pass_count:
                    done = True
                else:
                    pairing_names, pairing_count, uuids, node_list = self.rank_zero_stuff(tree, pharma_sites, node_list)
                    chunks, node_transmissions = self.data_distribution(pairing_names, node_list)
            # broadcast the complete list of nodes and names to the other nodes.
            done = comm.bcast(done, root=0)
        t2 = time()
        self.time = t2 - t1
        # collect all the nodes and write some fancy stuff.
        if MPIrank == 0:
            return pharma_sites.values(), pharma_sites.keys() 
        return None, None
   
    def collect_broadcast_dictionary(self, dict):
        """Collects all the elements of a dictionary from each node, combines
        them in one big dictionary, then broadcasts that dictionary to all nodes
        """
        empty = {}
        senddic = comm.gather(dict, root=0)
        [empty.update(i) for i in comm.bcast(senddic, root=0)]
        return empty
        #if MPIrank==0:
        #    senddic = dict 
        #else:
        #    senddic = None
        #if MPIrank == 0:
        #    for i in range(1,MPIsize):
        #        tempdic = comm.recv(source=i, tag=i)
        #        senddic.update(tempdic)
        #else:
        #    comm.send(dict, dest=0, tag=MPIrank)
        #return comm.bcast(senddic, root=0)

    def collect_recieve(self, node_transmissions, uuids):
        """Send active sites on this node that are not in pairings, collect
        active sites which are in pairings but not on this node."""
        for n_from, name, n_to in node_transmissions:
            tag_id = uuids[name]
            if MPIrank == n_from:
                sending = self._active_sites.pop(name)
                comm.send({name:sending}, dest=n_to, tag=tag_id)
            elif MPIrank == n_to:
                site = comm.recv(source=n_from, tag=tag_id)
                self._active_sites.update(site)
