import os, glob, shutil, json, pickle
import numpy as np
import matplotlib.pyplot as plt

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def save_dict2pkl(path, dictionary):

    with open(path,'wb') as fw:
        pickle.dump(dictionary, fw)

def load_pkl2dict(path):

    with open(path, 'rb') as fr:
        dictionary = pickle.load(fr)

    return dictionary

def min_max_norm(x):

    return (x - x.min() + (1e-30)) / (x.max() - x.min() + (1e-30))

def nhwc2nchw(x):

    return np.transpose(x, [0, 3, 1, 2])

def nchw2nhwc(x):

    return np.transpose(x, [0, 2, 3, 1])
