# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:56:41 2022

@author: nbrow
"""

import numpy as np 
from stl_tools import numpy2stl 


Image_Plot=np.load('Design_Files/UC_Design_Image_61.npy')
numpy2stl(Image_Plot, "UC_61.stl", scale=1e-5, mask_val = 1e-6)