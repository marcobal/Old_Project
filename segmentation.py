# -*- coding: utf-8 -*-
"""
-----------------
Segmentation Core
-----------------

Created on Mon Feb 17 16:03:57 2020

@author: Marco
"""

import utils
import numpy as np
from tqdm import tqdm
import math
import pygco
from skimage import measure
import skimage.morphology as morph
from enum import Enum, unique, auto
from sklearn.mixture import BayesianGaussianMixture
from scipy import special
from scipy import stats
from scipy.ndimage.morphology import binary_closing
import scipy.ndimage as nd

"""
UTILITIES
"""


@unique
class Type_Model(Enum):
    GAUSSIAN = auto()
    LN_GAUSSIAN = auto()
    GAUSSIAN_MIXTURE = auto()
    LOGNORMAL = auto()
    WEIBULL = auto()
    GAMMA = auto()
    INVERSE_GAMMA = auto()
    @staticmethod
    def to_list():
        return [element.name for element in Type_Model]    
    @staticmethod
    def to_string():
        return str([element.name for element in Type_Model])
    
    
"""
SEGMENTATION
"""


class Segmentation:
    
    def __init__(self):
        self.image = None
        self.image_raw = None
        self.mask = None
        self.pre_proc_image = None
        self.size_r = None
        self.size_c = None
        self.size_d = None
        # Graph
        self.seed = None
        self.seeds = []
        self.unique_levels = None
        # Membership
        self.membership = None
        self.membership_unprocessed = None
        self.membership_union = None
        # Models
        self.type_model = None
        self.num_param = None
        self.func_model_estimation = None
        self.func_model = None
        self.model = None
        self.likelihood = None
        self.likelihoods = []
        # Spatial model
        self.spatial_centroids = []
        self.compactness_indices = []
        self.distance_unary = None
        # Graph Cut
        self.initialization = None
        # Result
        self.segmented_image = None
        # Options
        self.is_filter_with_compact_index = True
        self.is_binary_membership = True
        self.is_auto_threshold_membership = False
        self.threshold_membership = 0
        self.threshold_membership_auto = 0
        self.gc_max_iter = 100
        self.gc_pairwise_weight = 0.1
        self.is_local_term_in_unary = True
        self.is_filter_likelihood_with_membership=True
        self.neighbourhood = []
        self.closing_sphere_radius = 1
        self.scaling_factor = 0
        self.scaled_sphere_radius = 0
        self.mask_thr = 0
        self.seeds_number = 0
        self.mean = 0
        self.means = []
        self.likelihood_scaling_factor = 0
        
        
    def set_image(self, image, mask = None, image_raw=None):
        self.mask = mask
        self.image = utils.normalize_datacube_0_1(image)*255
        self.size_r, self.size_c , self.size_d = np.shape(image)
        if image_raw is None:
            self.image_raw = image
        else:
            if np.shape(image_raw) != np.shape(image):
                raise ValueError("The raw image shape must match.")
            self.image_raw = image_raw
        
        
    def set_model_type(self, type_model):
        if type_model == Type_Model.GAUSSIAN:
            self.type_model = Type_Model.GAUSSIAN
            self.num_param = 2
            self.func_model_estimation = self._gaussian_model_estimation
            self.func_model = utils.gaussian_model
        elif type_model == Type_Model.LN_GAUSSIAN:
            self.type_model = Type_Model.LN_GAUSSIAN
            self.num_param = 2
            self.func_model_estimation = self._gaussian_model_estimation
            self.func_model = utils.ln_gaussian_model
        elif type_model == Type_Model.GAUSSIAN_MIXTURE:
            self.type_model = Type_Model.GAUSSIAN_MIXTURE
            self.num_param = 3
            self.func_model_estimation = self._gaussian_mixture_model_estimation
            self.func_model = None
        elif type_model == Type_Model.LOGNORMAL:
            self.type_model = Type_Model.LOGNORMAL
            self.num_param = 2
            self.func_model_estimation = self._lognormal_model_estimation
            self.func_model = utils.lognormal_model
        elif type_model == Type_Model.WEIBULL:
            self.type_model = Type_Model.WEIBULL
            self.num_param = 2
            self.func_model_estimation = self._weibull_model_estimation
            self.func_model = utils.weibull_model
        elif type_model == Type_Model.INVERSE_GAMMA:
            self.type_model = Type_Model.INVERSE_GAMMA
            self.num_param = 2
            self.func_model_estimation = self._gamma_model_estimation
            self.func_model = utils.inverse_gamma_model
        else:
            raise ValueError("No support for model {0}. Supported types are: \
                             {1}".format(type_model.name, Type_Model.to_string()))
        
        
    def configure_options(self, 
                          is_filter_with_compact_index=True,
                          threshold_membership=0.7, 
                          is_binary_membership=True,
                          is_auto_threshold_membership=False, 
                          gc_max_iter=100, gc_pairwise_weight=0.1,
                          is_local_term_in_unary=True,
                          is_filter_likelihood_with_membership=True,
                          neighbourhood = [],
                          mask_thr = 0,
                          closingSphereRadius = 1,
                          seeds_number = 0,
                          likelihood_scaling_factor = 0):
        self.is_filter_with_compact_index = is_filter_with_compact_index
        self.threshold_membership = threshold_membership
        self.is_auto_threshold_membership = is_auto_threshold_membership
        self.gc_max_iter = gc_max_iter
        self.gc_pairwise_weight = gc_pairwise_weight
        self.is_local_term_in_unary = is_local_term_in_unary
        self.is_filter_likelihood_with_membership = is_filter_likelihood_with_membership
        self.neighbourhood = neighbourhood
        self.closing_sphere_radius = closingSphereRadius
        self.scaled_sphere_radius = closingSphereRadius
        self.is_binary_membership = is_binary_membership
        self.is_filter_likelihood_with_membership = is_filter_likelihood_with_membership
        self.mask_thr = mask_thr
        self.seeds_number = seeds_number
        self.likelihood_scaling_factor = likelihood_scaling_factor
            
    
    def reset_flood_fill(self):
        self.weight_images = []
        self.unique_levels = []
        self.cost_images = []
        self.propagations = []
        self.membership = []
        self.membership_unprocessed = []
        
    def reset_models(self):
        self.models = []
        self.likelihoods = []

    def reset_graph_cut(self):
        self.D = None
        self.V = None
        
    def reset_all(self):
        self._reset_flood_fill()
        self._reset_models()
        self._reset_graph_cut()
        
    def _compute_compact_index(self, area, perimeter):
        return (area * 4*math.pi) / (perimeter * perimeter)
    
    
    def _is_valid_prop(self, properties):
        if len(properties) == 1 and 30 < properties[0].area < 0.5*(self.size_r*self.size_c):
            return True
        return False
    
    
    def _bin_and_morph_membership(self, membership):
        closed_memb_mask = binary_closing(membership)
        c_memb = np.zeros(np.shape(membership))
        c_memb[closed_memb_mask] = 1
        return c_memb
        
        
    def _optimize_threshold_membership(self, membership):
        m = membership.copy()
        thresholds = np.linspace(80, 70, 80-69, endpoint=True)/100
        ci = np.zeros([len(thresholds)])
        for idx, t in enumerate(thresholds):
            m_b = m.copy()
            m_b[m>=t] = 1
            m_b[m<t] = 0            #since voxels outside mask have negative membership
                                    #they will be set to 0 in m_b and therefore ignored 
            m_b = m_b.astype('int8')
            m_b = self._bin_and_morph_membership(m_b)
            properties = measure.regionprops(m_b.astype('int8'))
            if self._is_valid_prop(properties):
                ci[idx] = self._compute_compact_index(properties[0].area, properties[0].perimeter)
        best_idx = np.argmax(ci)
        best_threshold = thresholds[best_idx]
        self.threshold_membership_auto = best_threshold
        return best_threshold
            
    
    
    def _flood_fill_core(self):
        
        seed = self.seed
        seed_value = self.image_raw[seed]
        # Determine the unique level for a seed        
        rawValuesWithinMask = self.image_raw[self.mask == 1]
        
        diffs = np.absolute(rawValuesWithinMask - seed_value)
        unique_lev = np.unique(diffs)                        
        self.unique_levels = np.copy(unique_lev)
        
        #Convert the membership threshold into the corresponding 
        #maximum tolerance value for the flood_fill
        
        maxDiff = np.amax(unique_lev)
        
        diff = np.amax(rawValuesWithinMask) - seed_value
        tol = (1-self.threshold_membership)*diff
        
        maskBoundedImage = np.copy(self.image_raw)

        if self.is_binary_membership and not self.is_filter_likelihood_with_membership: #No need to compute different levels of memebership

            #Set voxels value for which mask == 0 to a value high enough to prevent flood_fill
            #from propagating outside the lung thus reducing computational overhead       
            maskBoundedImage[self.mask == 0] = maskBoundedImage[seed] + tol + 1
            maskBoundedImage = np.append(maskBoundedImage,np.zeros((1,self.size_c,self.size_d)),axis=0)

            #Compute binary membership by finding all voxels connected with seed and 
            #having tolerance<=tol
            memb = morph.flood_fill(maskBoundedImage[0:self.size_r,:,:],seed,9999,selem = self.neighbourhood,tolerance = tol)            
            memb[memb != 9999] = 0
            memb[memb == 9999] = 1

        else: #Need to compute different levels of membership

            cost_img = np.zeros_like(self.image_raw) - 1
            
            #Set voxels value for which mask == 0 to a value high enough to prevent any 
            #iteration of the flood_fill from propagating outside the lung thus reducing computational overhead
            maskBoundedImage[self.mask == 0] =  maskBoundedImage[seed] + maxDiff + 1
            maskBoundedImage = np.append(maskBoundedImage,np.zeros((1,self.size_c,self.size_d)),axis=0)
     
        
            #If we don't need the membership of all the voxels in the lung but only
            #the membership of the voxels in the region associated to a seed
            if not self.is_filter_likelihood_with_membership and not self.is_auto_threshold_membership:
                unique_lev = [j for i,j in enumerate(unique_lev) if j <= tol]
    
            #Find regions connected with seed for each tolerance level contained in unique_lev 
            for idx, level in enumerate(tqdm(reversed(unique_lev), total=len(unique_lev))):
                iterCost = morph.flood_fill(maskBoundedImage[0:self.size_r,:,:],seed,9999,selem = self.neighbourhood,tolerance = level)            
                cost_img[iterCost == 9999] = level

            # Membership computation
            norm_cost = cost_img/maxDiff
            memb = np.ones_like(norm_cost) - norm_cost
            memb[memb > 1] = -1 #Voxels with memb>1 are voxels outside the mask so set them to -1
            self.membership_unprocessed = np.copy(memb) 
            
            if self.is_auto_threshold_membership:
                threshold = self._optimize_threshold_membership(memb)
                memb[memb<threshold] = 0
            else:
                memb[memb<self.threshold_membership] = 0      
 
            if self.is_binary_membership:
                memb[memb != 0] = 1
     
        self.membership = memb        

        
    def _create_mask(self):
        
        print("Creating mask...")
               
        #Thresholding
        thrImage = np.zeros_like(self.image_raw)
        thrImage[self.image_raw <= self.mask_thr] = 1
        thrImage[self.image_raw > self.mask_thr] = 0    
        thrImage = thrImage.astype('int8')
        thrImage_copy = np.append(thrImage,np.zeros((1,self.size_c,self.size_d)),axis=0)

        
        #Find the image background
        point = (self.size_r-1,self.size_c-1,self.size_d-1)        
        outside_lung = morph.flood_fill(thrImage_copy[0:self.size_r,:,:],point,-1,selem = morph.cube(3),tolerance = 0)                             
        
        #Set the image background as "outside of the lung" (i.e. to 0)
        out = thrImage
        out[outside_lung == -1] = 0

        
        #Find a non-background voxel surely belonging to the lung 
        temp = nd.minimum_filter(thrImage,size = 4)
        xx, yy = np.meshgrid(range(self.size_r),range(self.size_c),indexing = 'ij')
                
        z_seed = np.int16(self.size_d/2)
        y_arr = yy[temp[:,:,z_seed] == 1]
        x_arr = xx[temp[:,:,z_seed] == 1]
        
        idx = np.argmax(y_arr)
        y_seed = y_arr[idx]
        x_seed = x_arr[idx]
        
        starting_point = (x_seed,y_seed,z_seed)
        
        
        #Get all voxels belonging to the lung and connected
        main_lung_region = morph.flood_fill(thrImage_copy[0:self.size_r,:,:],starting_point,-1,selem = morph.cube(3),tolerance = 0)                     

        #Set to 0 voxels that are non-background but do not belong to the lung either
        #The final result is an extended set containing connected and non-connected voxels 
        #belonging to the lung         
        aux = np.copy(thrImage)
        aux[main_lung_region == -1] = 0
        aux[:,0:np.int16(self.size_c/2),:] = 0
        
        out[aux == 1] = 0
        

        #Some wrongly classified voxels (set to 1 but outside the lung) may still appear 
        #near the lung
        #To prevent them from affecting the closing and wrongly extend the final mask, a first 
        #closing is performed with a struct. element very narrow along x and y but extended along z
        #In this way we can connect almost all the voxels belonging to the lung leaving all the 
        #others outside
        sel_em_slice = np.array([[0,1,0],[1,1,1],[0,1,0]]).reshape((3,3,1))
        sel_em = np.broadcast_to(sel_em_slice,(3,3,13))
        semi_closed = morph.closing(out,selem = sel_em)
        
        
        #Now a new flood-fill is performed that will find a more complete set of connected
        #voxels belongong to the lung
        semi_closed = np.append(semi_closed,np.zeros((1,self.size_c,self.size_d)),axis=0)
        semi_closed_lung = morph.flood_fill(semi_closed[0:self.size_r,:,:],starting_point,-1,selem = morph.cube(3),tolerance = 0)         
        semi_closed_lung[semi_closed_lung != -1] = 0
        semi_closed_lung[semi_closed_lung == -1] = 1

        #Scaling the size with subsampling to make the final closing faster          
        scaledShape = []       
        
        if self.scaling_factor > 1:
            range_x = range(0,self.size_r,self.scaling_factor)
            range_y = range(0,self.size_c,self.scaling_factor)
            range_z = range(0,self.size_d,self.scaling_factor)
            xxx_scaled, yyy_scaled, zzz_scaled = np.meshgrid(range_x,range_y,range_z,indexing = 'ij')
        
        
            scaledShape = [int(val/self.scaling_factor) for i,val in enumerate((self.image).shape)]
            scaledMask = semi_closed_lung[xxx_scaled.flatten(),yyy_scaled.flatten(),zzz_scaled.flatten()]
            scaledMask = np.reshape(scaledMask,scaledShape)
        else:
            scaledMask = semi_closed_lung
            scaledShape = semi_closed_lung.shape            
        
        
        #Adding external slices to the mask to prevent dilation from exceeding the borders           
        increase = int(2*self.scaled_sphere_radius + 2)
        enlarged_size = [val + 2*increase for i,val in enumerate(scaledShape)]
        enlarge_for_closing = np.zeros(enlarged_size).astype('int8')
        enlarge_for_closing[increase:-increase,increase:-increase,increase:-increase] = scaledMask
        
        #Closing
        closedMask = morph.closing(enlarge_for_closing,morph.ball(self.scaled_sphere_radius))
        closedMask = closedMask[increase:-increase,increase:-increase,increase:-increase]
        
        #Restoring original size of the mask
        resizedMask = nd.zoom(closedMask.astype('float64'),self.scaling_factor,prefilter=False)
        
        #Binarizing mask
        normResizedMask = utils.normalize_datacube_0_1(resizedMask)     
        normResizedMask = np.rint(normResizedMask).astype('int8')
        normResizedMask = morph.erosion(normResizedMask,morph.ball(3))
        
        self.mask = normResizedMask
        
    def _adapt_size_for_mask_matching(self):

        #Prevents the mismatch between the size of the image and the size of the mask
        
        print("Adapting image size...")
        
        if self.mask is None: #Mask has to be created
            
            #Compute the scaling_factor that will be used for the creation of the mask
            while self.scaled_sphere_radius >= 7:
                self.scaling_factor += 1 
                self.scaled_sphere_radius = np.ceil(self.closing_sphere_radius/self.scaling_factor)
            img_size = np.array(np.shape(self.image))
            
            #Determine new sizes so that they are integer multiples of the scaling_factor
            self.size_r, self.size_c , self.size_d = (np.floor(img_size/self.scaling_factor)*self.scaling_factor).astype('int16')

        else: #Mask is already available
            self.size_r, self.size_c , self.size_d = np.shape(self.mask)
        
        #Adapt data to the new size cutting some slices
        self.image = self.image[0:self.size_r,0:self.size_c,0:self.size_d]
        self.image_raw = self.image_raw[0:self.size_r,0:self.size_c,0:self.size_d]
        
    def _flood_fill(self):
        [self._flood_fill_core(seed) for seed in self.seeds]

        
    def _filter_membership(self):

        
        #Discard seeds whose associated region is not composed by at least 1000 voxels
        
        if np.sum(self.membership) <= 1000:
            return True
        
        #Discard seeds whose associated region has a too high standard deviation
        
        aux = np.copy(self.membership)
        aux[aux>0] = 1
        aux = aux.astype('int8')
       
        membValues = (self.image_raw)[aux == 1]
        
        if not self.is_auto_threshold_membership: 
            thr = self.threshold_membership
        else:
            thr = self.threshold_membership_auto
                
        
        valuesInMask = self.image_raw[self.mask == 1]
        seed_value = self.image_raw[self.seed]
        diff = np.amax(valuesInMask) - seed_value
        tol = (1-thr)*diff
        
        upperBound = np.amin([np.amax(valuesInMask),seed_value + tol])
        lowerBound = np.amax([np.amin(valuesInMask),seed_value - tol])
        
        totalRange = upperBound - lowerBound
        
        norm_std = np.sqrt(np.var(membValues))/totalRange
        print(norm_std)
        if norm_std > 0.3: return True
        
        print("samples: " + str(np.sum(self.membership)))
        
        return False
                    
       
    def _gaussian_model_estimation(self, membership):
        weighted_image = np.multiply(self.image_raw, membership)
        mean = np.sum(weighted_image.flatten()) / np.sum(membership.flatten())
        var = np.sqrt(np.sum(np.square(weighted_image.flatten()-mean)) / np.sum(membership.flatten()))
        return np.asarray([mean, var])    
    
    def _gaussian_mixture_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        estimator = BayesianGaussianMixture()
        samples = self.image_raw[binMemb == 1].flatten()
        estimator.fit(np.reshape(samples, [len(samples),1]))
        return estimator    
    
    def _lognormal_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_raw, binMemb)
        samples = filteredImage[filteredImage>0].flatten()
        k1 = np.sum(np.log(samples)) / np.sum(membership.flatten())
        k2 = np.sum(np.square(np.log(samples)-k1)) / np.sum(membership.flatten())
        return np.asarray([k1, k2])
    
    def _weibull_model_estimation(self, membership):
        binMemb = np.zeros_like(membership)
        binMemb[membership>0] = 1 
        filteredImage = np.multiply(self.image_raw, binMemb)
        samples = filteredImage[filteredImage>0].flatten()
        k1 = np.sum(np.log(samples)) / np.sum(membership.flatten())
        k2 = np.sum(np.square(np.log(samples)-k1)) / np.sum(membership.flatten())
        aux_mu = k1 - ((np.sqrt(k2)*special.polygamma(0, 1))/(np.sqrt(special.polygamma(1, 1))))
        mu = np.exp(aux_mu)
        eta = np.sqrt(special.polygamma(1, 1) / k2)
        return np.asarray([eta, mu])
    
    def _gamma_model_estimation(self, membership):
        
        if self.type_model == Type_Model.INVERSE_GAMMA:             
            #To obtain the inverse_gamma model, a normal gamma estimation is performed
            #on the reversed samples 
            vals = -((self.image_raw[membership > 0]).flatten())
        else:
            vals = (self.image_raw[membership > 0]).flatten()
        L, location, scale = stats.gamma.fit(vals,loc = np.amin(vals))
        mu = stats.gamma.mean(L,location,scale)
        
        return np.asarray([L, mu, location])
    
       
    def _estimate_model(self):
            single_model = self.func_model_estimation(self.membership)
            self.model = single_model
            

    def _estimate_spatial_models(self):
        for m in self.membership:
            m[m>0] = 1
            m = m.astype('int8')
            props = measure.regionprops(m)
            self.spatial_centroids.append(props[0].centroid)
            ci = self._compute_compact_index(props[0].area,props[0].perimeter)
            self.compactness_indices.append(ci) 
            
            
    def _compute_likelihood(self):
        model_params = self.model
        if self.type_model != Type_Model.GAUSSIAN_MIXTURE:
            likelihood = self.func_model(self.image_raw[self.mask == 1], model_params)
        else:
            image_flatten = (self.image_raw.copy())[self.mask == 1]
            image_flatten = np.reshape(image_flatten, [len(image_flatten), 1])
            likelihood = model_params.score_samples(image_flatten)
            likelihood = likelihood.flatten()
            self.mean = np.sum(np.multiply(model_params.means_, model_params.weights_.reshape((-1,1))))

        if self.is_filter_likelihood_with_membership:
            memb = self.membership_unprocessed[self.mask == 1]
            likelihood = np.multiply(likelihood, memb)
        
        self.likelihood = likelihood
            
    
    def _compute_distances_unary(self):
        self.distance_unary = np.zeros([*np.shape(self.image), len(self.seeds)])
        for idx, centroid in enumerate(self.spatial_centroids):
            rr, cc = np.meshgrid(range(self.size_r), range(self.size_c), indexing='ij')
            d_squared = np.square((rr[:,:] - centroid[0]))+np.square(cc[:,:] - centroid[1])
            self.distance_unary[:,:,idx] = np.sqrt(d_squared)
            
    
    def _run_graph_cut(self):
                 

        likelihood = np.asarray(self.likelihoods)
        likelihood = likelihood*(self.likelihood_scaling_factor)

       
        #Creating background label:
        
        #If at least one label (i.e. seed) has non-zero p(x/label) for a sample x, the
        #background is set to disfavor the choice of the background label
        bg = np.zeros_like(likelihood[0,:])
        bg[np.isneginf(np.amax(likelihood,axis = 0)) == True] = 5       
        
        
        #If each label has zero p(x/label) for a sample x, background is set in a way
        #so that the background label will be chosen mandatorily
        shape = likelihood.shape
        copy = np.copy(likelihood).reshape((1,len(likelihood.flatten())))
        copy[np.isneginf(copy) == True] = np.inf
        copy = np.reshape(copy,shape)      
        idx = np.isinf(np.amin(copy,axis = 0)) == False
        bg[idx] = np.amin(copy,axis = 0)[idx] - 10   
        
    
        #Prepare the likelihood matrix for the graph-cut
        likelihood = np.append(likelihood,np.reshape(bg,(1,len(bg))),axis = 0).transpose().astype('int32', order = 'C')        
        
        
        num_labels = likelihood.shape[1]        
        
        #Prepare the pairwise costs
        V = np.ndarray(shape=(num_labels, num_labels)).astype('int32',order = 'C')
        for l1 in range(0, num_labels):
            for l2 in range(0, num_labels):
                if l1 == l2:
                    V[l1, l2] = 0
                elif l1 != num_labels - 1 and l2 != num_labels - 1 and np.absolute(self.means[l1]-self.means[l2]) < 250:
                    V[l1, l2] = self.gc_pairwise_weight + 1
                else:
                    V[l1, l2] = self.gc_pairwise_weight
                    

    
        #Prepare matrix of all edges between voxels inside the lung
        print("Creating graph edges...")
        
        x_edges = utils.edges_from_mask(self.mask,0)
        y_edges = utils.edges_from_mask(self.mask,1)
        z_edges = utils.edges_from_mask(self.mask,2)
        
        edges = np.append(np.append(x_edges,y_edges,axis = 0),z_edges,axis = 0).copy(order = 'C')
  
      
        #Run graph-cut  
        print("Running graph-cut...")
        result = pygco.cut_from_graph(edges,-likelihood,V,algorithm = 'swap')


        #Build the 3d image with labelled lung
        result_image = np.ones_like(self.mask)*(num_labels) #set a different label for
                                                            #pixels outside the lung
        result_image[self.mask == 1] = result
        
        
        self.segmented_image = result_image

          
    def _add_seed(self):
        
        if self.membership_union is None:            
            self.membership_union = 1 - self.mask 


        xxx, yyy, zzz = np.meshgrid(range(self.size_r),range(self.size_c),range(self.size_d),indexing = 'ij')
        
        x = xxx[self.membership_union == 0]
        y = yyy[self.membership_union == 0]
        z = zzz[self.membership_union == 0]

        idx_rand = np.random.randint(0,len(x)-1)
        
        seed = (x[idx_rand],y[idx_rand],z[idx_rand])
        self.seed = seed
        
    
    def _likelihood_from_seeds(self):

        print("Estimating models and computing likelihood...")  

        attempt_number = 0
        t = tqdm(total = self.seeds_number)
        while len(self.seeds) != self.seeds_number:
            
            if attempt_number == 15:
                t.update(self.seeds_number - len(self.seeds))
                break
            
            self._add_seed()
            
            
            self._flood_fill_core()

            if self._filter_membership():
                attempt_number += 1
                continue   
            
            self._estimate_model()
            
            
            attempt_number = 0
            self._compute_likelihood()            
            self.likelihoods.append(self.likelihood)
            self.membership_union[self.membership != 0] = 1
            self.means.append(self.mean)
            self.seeds.append(self.seed)            
            t.update(1)
    
    def run(self):
        if self.image is None:
            raise ValueError("Set the image before running the segmentation.")
        #if len(self.seeds) == 0:
        #    raise ValueError("Add at least one seed before running the segmentation.")
        if self.type_model is None:
            raise ValueError("Set the model before running the segmentation.")
            
  ##Preparation        
        self._adapt_size_for_mask_matching()
        if self.mask is None:
            self._create_mask()     

  #Automatic seed setting, initial ROI extraction, model estimation and unary term computation 
        self._likelihood_from_seeds()
        
  ##Graph edges computation, pairwise terms definition and graph-cut energy minimization         
        self._run_graph_cut()
        
        
    def get_segmentation(self):
        if self.segmented_image is not None:
            return self.segmented_image
        else:
            raise ValueError("Run the segmentation method first.")
