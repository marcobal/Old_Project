# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:28:31 2020

@author: Marco
"""

from segmentation import Segmentation, Type_Model
import utils
import pathlib
import numpy as np
import medpy.io as io


neighbourhood = np.array([[[0,0,0],
[0,1,0],
[0,0,0]],
[[0,1,0],
[1,1,1],
[0,1,0]],
[[0,0,0],
[0,1,0],
[0,0,0]]])

model = Type_Model.GAUSSIAN_MIXTURE
#model = Type_Model.INVERSE_GAMMA
#model = Type_Model.LN_GAUSSIAN

"""
LUNG BEGIN
"""
obj = "LUNG"
num = 3

if num < 10:
    paz = "00" + str(num)
else:
    paz = "0" + str(num)

if num <= 10:
    fileName = str("coronacases_org_" + paz + ".nii")
else:
    fileName = str("radiopaedia_" + paz + ".nii")

"""
LUNG END
"""

"""
WRIST BEGIN
"""
    
#obj = "WRIST"
#paz = "057"
#fileName = "WRIST.mha"
    
"""
WRIST END
"""


#Load 3d image
path = pathlib.Path("data") / str("_" + obj + "_VOLUMES") / str("paz" + paz) / fileName
image3d = io.load(str(path))[0]



#Check if mask already exists, if not, load the seed point to create the new mask 
mask = None;
maskPath = path.parents[0] / "mask_improved.nii"

if maskPath.exists(): 
    mask = io.load(str(maskPath))[0]

if num <= 10:
    image3d_pre_proc = utils.image_preprocessing(image3d,[np.amin(image3d),500])
else:
    image3d_pre_proc = image3d



"""
Segmentation
"""
segm = Segmentation()

#no auto_threshold if binary_membership is true
segm.set_image(image3d_pre_proc, mask, image_raw=image3d_pre_proc)
segm.configure_options(threshold_membership= 0.7,
                       is_binary_membership=True,
                       is_auto_threshold_membership=False,
                       gc_max_iter=100,
                       gc_pairwise_weight= 3,
                       is_filter_likelihood_with_membership = False,
                       neighbourhood = neighbourhood,
                       mask_thr = -500, #for initial image binarisation (required for the creation of the mask)
                       closingSphereRadius = 21,
                       seeds_number = 7,
                       likelihood_scaling_factor = 0.75)

segm.set_model_type(model)

segm.run()


#Save the new computed mask if no mask already exists
if not(maskPath.exists()):
    io.save(segm.mask,str(maskPath))


#Save segmentation result
result = segm.segmented_image
io.save(result,str(path.parents[0] / "result.nii"))

if obj == "LUNG":
    
    if num <= 10:
        mean_thr = -600
        utils.reduce_oversegm(result,image3d,mean_thr)
        io.save(result,str(path.parents[0] / "result_imp.nii"))
        
        test_image3d = io.load(str(path.parents[0] / str("coronacases_" + paz + "_inf_mask.nii")))[0]
    else:
        test_image3d = io.load(str(path.parents[0] / str("radiopaedia_" + paz + "_inf_mask.nii")))[0]


#Evaluate result quality
eval_result_path = path.parents[0] / str("test_results_" + paz + ".txt")
output_file = open(eval_result_path, mode = 'w')
    
utils.evaluate_and_print_results(test_image3d,result,output_file)

output_file.close()

