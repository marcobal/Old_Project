# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:07:29 2020

@author: Marco
"""
import warnings
import numpy as np
import math
from scipy import special
from tabulate import tabulate


#--- GRAPH


def flood_fill_core(original, weights, levelsCurve, seed, level):
   	if level < weights[seed[0], seed[1]]: # it operates only if the weight is larger than the current level
   		return original
   
   	queue = [seed]; # list of points to be assigned a level
   	while queue:
   		current = queue.pop(0)
   		x = current[0]
   		y = current[1]
   		original[x][y]=255
   		weights[x][y]=999
   		levelsCurve[x][y]=level
   
   		for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)): # compute the 4-neighbours
   			if 0<=x2<=weights.shape[0]-1 and 0<=y2<=weights.shape[1]-1:
   				if level >= weights[x2, y2]:
   					queue.append( (x2,y2) ) # append and iterate to this point
   					original[x2][y2]=255
   					weights[x2][y2]=999
   					levelsCurve[x][y]=level
   	return original


#--- IMAGE   
 
    
def normalize_datacube_0_1(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    if array.dtype != np.float64:
        array = np.copy(array.astype(np.float64))
#    n_dim = np.ndim(array)
#    if n_dim == 3:
#        bands = np.shape(array)[2]
#        for b in range(0, bands):
#            band = array[:,:,b]
#            band_min, band_max = band.min(), band.max()
#            array[:,:,b] = ((band - band_min)/(band_max - band_min))
#        return array
#    elif n_dim == 2:
#        array_min, array_max = array.min(), array.max()
#        return ((array - array_min)/(array_max - array_min))
#    else:
#        raise ValueError("Expected array dimension to be 2 or 3. Received {}.".format(n_dim))
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))


#--- PARAMETERS APPROXIMATION


def f(L, k2):
    """k2 - polygamma for L estimation"""
    return (k2-special.polygamma(1, L))


#--- MODELS  
      
        
def gaussian_model(array, params):
    mean, var = params
    dev_std = math.sqrt(var)
    A = 1 / (dev_std * math.sqrt(2*math.pi))
    E = -0.5 * np.square((array - mean) / dev_std)
    return A * np.exp(E)


def ln_gaussian_model(array, params):
    mean, var = params
    A = -0.5*math.log(2*math.pi)
    B = -0.5*math.log(var)
    C = -(1/(2*var))*np.square(array-mean)
    return A + B + C


def lognormal_model(array, params):
    k1, k2 = params
    dev_std = math.sqrt(k2)
    A = 1 / (dev_std * array * math.sqrt(2*math.pi))
    E = -0.5 * (np.square(np.log(array) - k1) / k2)
    return np.log(A * np.exp(E))


def weibull_model(array, params):
    array = array + 0.1
    eta, mu = params
    A = eta * np.power(array,eta-1) / np.power(mu,eta)
    E = np.power(array/mu, eta)
    return np.log(A * np.exp(- E))

def compute_ln_gamma_function(shape_factor):
    
    #Computes gamma_function using iteratively the property 
    #gamma_function(L) = (L-1)*gamma_function(L-1)
    
    last_iter = np.floor(shape_factor).astype('int32')
    last_term = np.log(special.gamma(shape_factor - last_iter))
    out = last_term
    
    for i in range(1,last_iter+1):
        out += np.log(shape_factor - i)
        
    return out

def inverse_gamma_model(array, params):
    
    warnings.filterwarnings("ignore", category = RuntimeWarning)
    
    array = array + 0.1
    L, mu, transl = params
    A = -compute_ln_gamma_function(L)
    B = L * np.log(L/(mu-transl))
    C = (L-1)*np.log(-array - transl)
    E = -L * (-array - transl) / (mu-transl)
    out = A + B + C + E
    out[array > -transl] = np.NINF
    return out

def edges_from_mask(mask, axis):
    
    #Finds all edges along given axis for the voxels inside the lung (i.e. for which mask == 0)
    
    shape = mask.shape
    voxels_inside_mask = np.sum(mask.flatten())
        
    idx_image = np.zeros_like(mask).astype('int32')
    idx_image[mask == 1] = np.arange(1,voxels_inside_mask+1)
    transl_idx_image = np.zeros_like(idx_image)

    if axis == 0:
        transl_idx_image[0:shape[0]-1,:,:] = idx_image[1:shape[0],:,:]
    elif axis == 1:
        transl_idx_image[:,0:shape[1]-1,:] = idx_image[:,1:shape[1],:]
    elif axis == 2:
        transl_idx_image[:,:,0:shape[2]-1] = idx_image[:,:,1:shape[2]]
        
    prod = np.multiply(idx_image,transl_idx_image)
        
    node_a = idx_image[prod != 0] -1
    node_b = transl_idx_image[prod != 0] -1
    
    edges = np.append(node_a.reshape((-1,1)),node_b.reshape((-1,1)),axis=1)
    
    return edges

def evaluate_and_print_results(test_map,segm_output,out_file):
    
    labels = np.unique(segm_output).tolist()
    segm_size = segm_output.shape
    test_map_copy = np.copy(test_map.astype('int8'))[0:segm_size[0],0:segm_size[1],0:segm_size[2]]
    total = np.sum(test_map_copy == 1)
#    aux = np.copy(test_map_copy)
#    aux_ext = np.append(aux,np.zeros((1,segm_size[1],segm_size[2])),axis = 0)
#    xxx, yyy, zzz = np.meshgrid(range(segm_size[0]),range(segm_size[1]),range(segm_size[2]),indexing = 'ij')
#    
#    counts = []
#    t = tqdm(total=np.sum(aux != 0))
#    while not np.sum(aux) == 0:
#        
#        x = xxx[aux != 0]
#        y = yyy[aux != 0]
#        z = zzz[aux != 0]
#        
#        test_area = morph.flood_fill(aux_ext[0:segm_size[0],:,:],(x[0],y[0],z[0]),2,selem = morph.ball(1),tolerance = 0)
#        
#        count_for_test_area = [np.sum(np.logical_and(test_area == 2,segm_output == val)) for i,val in enumerate(labels)]
#        counts.append(np.asarray(count_for_test_area))
#        
#        aux[test_area == 2] = 0 
#        t.update(np.sum(test_area == 2))
#        
#    t.close()
        
    test_area_count = ["Infection"] 
    non_test_count = ["No-Infection"]
    precision_ratios = ["Precision"]
    accuracy_ratios = ["Accuracy"]
    
    for i,val in enumerate(labels):
        test_area_count.append(np.sum(np.logical_and(test_map_copy == 1,segm_output == val)))
        non_test_count.append(np.sum(np.logical_and(test_map_copy != 1,segm_output == val)))
        precision_ratios.append(test_area_count[-1]/(test_area_count[-1] + non_test_count[-1]))
        accuracy_ratios.append(test_area_count[-1]/total)

        
    table = [test_area_count,non_test_count,precision_ratios,accuracy_ratios]
    labels.insert(0,"Labels:")

    out_file.write(tabulate(table,headers = labels,numalign="right"))
    
    #return np.append(np.asarray(test_area_count).reshape((1,-1)),np.asarray(non_test_count).reshape((1,-1)),axis = 0)

def reduce_oversegm(segm_result,image_raw,mean_thr):
    
    shape = segm_result.shape
    image_raw_copy = np.copy(image_raw)[0:shape[0],0:shape[1],0:shape[2]]
    a = np.unique(segm_result)
    labels = np.delete(a,-1,0)
    idx_joint_region = np.zeros_like(segm_result).astype('bool')
    for i,val in enumerate(reversed(labels)):
        idx_label = segm_result == val
        if np.mean(image_raw_copy[idx_label]) <= mean_thr:
            idx_joint_region = np.logical_or(idx_joint_region,idx_label)
            label_joint_region = val
    
    segm_result[idx_joint_region] = label_joint_region

#def image_preprocessing(path):
#    image3d = io.load(str(path))[0]
#    image3d = image3d.astype('float64')
#    cutPath = path.parents[0] / "cut.json"
#
#    if cutPath.exists():
#        json_file = str(cutPath)
#        with open(json_file) as json_f:
#            cut = json.load(json_f)
#        
#        image3d = image3d[cut["start_c"]:cut["end_c"],cut["start_r"]:cut["end_r"],cut["start_f"]:cut["end_f"]]
#    
#    outputImage = np.floor(normalize_datacube_0_1(image3d)*255)
#    
#    return outputImage
    
def image_preprocessing(raw_image3d,window):
    
    #Windowing
    
    if len(window) != 2:
        raise ValueError("The window must have an upper bound and a lower bound")
    elif window[0] >= window[1]:
        raise ValueError("The first bound must be smaller than the second")
        
    pre_proc_image = raw_image3d.copy()
    pre_proc_image[pre_proc_image < window[0]] = window[0]
    pre_proc_image[pre_proc_image > window[1]] = window[1]
    
    return pre_proc_image