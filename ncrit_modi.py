import pandas as pd
import numpy as np
import astropy.constants as cons
import astropy.units as u
from scipy.interpolate import interp1d
import os
from io import StringIO
import re

parts = ['H2', 'para-H2', 'ortho-H2', 'electrons', 'H', 'He', 'H+']
colpart_dict = dict(zip(np.arange(1,7), parts))


def escape_beta(tau, geometry='lvg'):
	if geometry == 'lvg':
		return (1-np.exp(-1*tau))/tau
	elif geometry == 'us':#uniform sphere
		return (3/4./tau - 3/8./tau**3 + np.exp(-2*tau)*(3/4./tau**2+3/8./tau**3))

def interp_collrates(Tkin, temps, collrates_alltrans):
    colrates = np.zeros(collrates_alltrans.shape[0])
    for i, collrate in enumerate(collrates_alltrans):
        finterp = interp1d(temps, collrate, bounds_error=False, fill_value=(collrate[0],collrate[1]))
        if Tkin>temps[-1] or Tkin<temps[0]:
            print("@ylin: Tkin outside of database temperature range of collisional rates! Use bound values...")
        colrates[i] = finterp(Tkin)
    return colrates # a list of collrates at different transitions 

class DataBlock:
    def __init__(self, lines):

        f = StringIO()
        self.colpart = colpart_dict[np.int(lines[1][:1])]
        self.ncoltrans = np.int(lines[3])
        self.ntemps = np.int(lines[5])
        temps = np.array(lines[7].split(),dtype=float)
        self.temps = temps

        ts = [str(t) for t in temps]
        f.write('trans up low '+' '.join(ts)+'\n')

        for i in range(9,len(lines)):
            f.write(lines[i])
            f.write('\n')
        f.seek(0)
        self.df = pd.read_csv(f, sep=r'\s+', header=0)

class HeadBlock:
    def __init__(self, lines):

        f = StringIO()

        self.nlevels = np.int(lines[1])
        colnames = [str(colname) for colname in lines[2][1:].split('+')]
        f.write(' '.join(colnames)+'\n')
        for i in range(3, len(lines)):
            f.write(lines[i])
            f.write('\n')
        f.seek(0)
        self.df = pd.read_csv(f, sep=r'\s+', header=0)
    
class MolData:
    """ A class to read in moldata from Leiden database """

    def __init__(self, filepath=None):
        
        if filepath is None:
            print("@ylin: init moldata instance [empty]...")
        else:
            print("@ylin: init moldata instance [%s]..."%filepath)
        
        self.filepath = filepath
        self.data_loaded = False
        self.df = {}
        #
        self.hd0 = None
        self.hd1 = None
        
        self.comments = ''
        self.molname = ''
        self.mass = -1
        
        
    def load_data(self):
        
        f = open(self.filepath, mode='r')
        lines = f.readlines()
        f.close()
        # cut \n tails
        lines = [_[:-1] for _ in lines]
        
        assert re.search('MOLECULE',lines[0], re.I), "@ylin: Perhaps need to check data format manually!"

        lineblocks = self.splitLines_new(lines)
        self.hd0 = HeadBlock(lineblocks[0])
        self.hd1 = HeadBlock(lineblocks[1])
        
        for i in range(len(lineblocks)-3):
        	self.df[i] = DataBlock(lineblocks[i+3])

        self.comments = ''
        self.molname = str(lines[1])
        self.mass = np.float(lines[3])
        self.ncollpart = np.int(lineblocks[2][1])
        self.data_loaded = True
        return self

    def splitLines_new(self, lines):
    

        nlevels = int(lines[5])    
        n_lines = [4, nlevels+7]
        ntrans_simul = int(lines[nlevels+8])

        i_l = nlevels+10+ntrans_simul
        n_lines.append(i_l)
        ncols = int(lines[i_l+1])

        assert re.search('NUMBER OF COLL',lines[i_l], re.I), "@ylin: Perhaps need to check data format manually!"


        skiplines = 0
        for iicol, icol in enumerate(range(ncols)):
            n_lines.append(i_l+2+iicol+skiplines)
            ntrans = int(lines[i_l+5+iicol+skiplines])
            skiplines = 8+ntrans+skiplines
        n_lines.append(n_lines[-1]+9+ntrans)
        n_lines = np.array(n_lines)
        lineblocks = []
        for i_l, nline in enumerate(n_lines[:-1]):
            lineblocks.append(lines[nline:n_lines[i_l+1]])
        #lineblocks.append(lines[n_lines[i_l+1]:])
        return lineblocks

    def getData(self, n):
    	return self.df[n]

def ncrit_cal(moldata, low, up):

    df1 = moldata.hd0.df
    cols1 = df1.columns
    i_lo = int(df1[cols1[0]].values[df1[cols1[3]] == low])
    i_up = int(df1[cols1[0]].values[df1[cols1[3]] == up])
    gu = df1[cols1[2]].values[df1[cols1[0]]==i_up]
    Eu = df1[cols1[1]].values[df1[cols1[0]]==i_up]


    df2 = moldata.hd1.df
    cols2 = df2.columns
    Aul = float(df2[cols2[3]].values[(df2[cols2[1]]==i_up)&(df2[cols2[2]]==i_lo)])

    df3 = {}
    df3 = moldata.df.copy()
    ncrit = {}
    for key in df3.keys():
        colpart = df3[key].colpart
        df3[colpart] = df3.pop(key)
        coll_df = df3[colpart].df
        cols3 = coll_df.columns

        #upward collisions
        cond = (coll_df['low'] == i_up)
        ulevels = coll_df['up'].values[cond]
        guj = df1[cols1[2]].values[df1[cols1[0]].isin(ulevels)]
        Ej = df1[cols1[1]].values[df1[cols1[0]].isin(ulevels)]
        coll_to_up = coll_df[cols3[3:]][cond]

        coll_to_lo = coll_df[cols3[3:]][coll_df['up'] == i_up]
        temps = df3[colpart].temps 

        colltolo = np.sum(interp_collrates(Tkin, temps, coll_to_lo.values))
        colltohi = np.sum(guj/gu*interp_collrates(Tkin, temps, coll_to_up.values)*np.exp((Eu-Ej)*efac/k/Tkin))
        ncrit[colpart] = Aul/(colltolo+colltohi)
    return ncrit	

if __name__ == "__main__":

    #constants Energy cm^-1 to cgs
    efac = (1*u.eV).cgs.value*0.000123986
    k = cons.k_B.cgs.value

    mole = 'ch3cn'
    #mole = 'ech3oh'
    mole = 'co'
    moldata_path = '/Users/yuxinlin/radex/data/ch3cn.dat'
    moldata_path = '/Users/yuxinlin/radex/data/12co.dat'

    #moldata_path = '/Users/yuxinlin/radex/data/e-ch3oh.dat'


    moldata = MolData(moldata_path)
    moldata = moldata.load_data()


    up = 7
    low = 6

    #for complex levels, string format e.g.
    #up = '9_0'
    #low = '8_0'

    Tkin = 50.
    tau = 0.1
    ncrit = ncrit_cal(moldata, low, up)
    for n in ncrit.items():
        print('Critical density of %s with %s is %.3e cm^-3\n'%(mole, n[0], n[1]))
        print('(Optical depth correction factor:%.3f)'%(escape_beta(tau)))

