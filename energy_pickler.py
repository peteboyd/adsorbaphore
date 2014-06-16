#!/usr/bin/env python

import sys
import pickle
import os
import random

def de_pickle(file):
    f = open(file, 'rb')
    print "opening pickle file..."
    p = pickle.load(f)
    print "done"
    return p

def get_pickle_file():
    pwd = os.getcwd()
    if len(sys.argv) == 2:
        if os.path.exists(os.path.join(pwd, sys.argv[1])):
            return sys.argv[1]
        else:
            print "Could not find %s in the current working directory!"%(sys.argv[1])
            sys.exit()

    elif len(sys.argv) == 1:
        files = os.listdir(pwd)
        dist_files = [i for i in files if i[-11:] == 'co2dist.pkl']
        if len(dist_files) == 0:
            print "Couldn't find any CO2 distribution files"
            sys.exit()

        elif len(dist_files) == 1:
            return dist_files[0]
        
        else:
            randchoice = random.choice(dist_files)
            print "Too many pickle files! choosing one at random: %s"%(randchoice)
            return randchoice 

def main():

    pickle_file = get_pickle_file()
    dic = de_pickle(pickle_file)
    # next take the nets, co2's and calculte the energy from fastmc


if __name__=="__main__":
    main()
