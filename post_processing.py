import numpy as np
from config_fap import Options
from StringIO import StringIO

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

