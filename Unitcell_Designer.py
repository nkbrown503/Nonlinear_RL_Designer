# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:39:01 2021

@author: nbrow
"""

import sys
import time
import math
import numpy as np 
from Abaqus_INP_writer import INP_writer
from Matrix_Transforms import * 
from Design_Functions import Bezier_Design, UnitCell_To_Metamaterial
from Palmetto_JobScript_Writer import JobScript_Writer
E_X=20 #Number of Elements Per Unit Cell in the X Direction (Doubled if Mirror='Sinlge or Mirror='Double')
E_Y=20 #Number of Elements Per Unit Cell in Y Direction (Doubled if Mirror='Double')
Mesh_Complexity=1; #The number of elements in the X and Y Direction used to define a single element produced by the Bezier Curves
Iterations=1000 #How many unit-cells do you want to produce   #Length of the Unitcell in the X Direction
Aspect_Ratio=3
Load_Type='C'   #'C' for compression    'T' for Tension
AR_Name=Aspect_Ratio
Strain=0.2 #What magnitude of force will be applied in Abaqus?
Tesselate=True   #Do you want to tesselate the unit-cell in the x and y direction
Fillet=True
Mirror='Double'  #'Double' for Mirror about X and Y Axes    'Single' for mirror about Y axis    'None' for no mirror 
if Mirror!='Single' and Mirror!='Double' and Mirror!='None':
    sys.exit("Please specify Mirror as either ''Single', 'Double', or 'None' (Check Spelling) ")

if Mirror=='Single' and Aspect_Ratio>1:
    Aspect_Ratio=float(Aspect_Ratio/2)
Overlap=True   #If overlap=True, the tesselate unit-cells will be translated a half unit-cell for each row, if overlap=False, the unit-cells will be uniformly stacked

Type='Corner'   # 'Corner': Material Defined at the four corners of the unit-cell  'Edge': Material defined at the top and bottom edges of the unit-cell

E_X*=Aspect_Ratio
E_X=int(E_X)
if Tesselate: # Adjust UX and UY for how many tesselations needed 
    UX=3
    UY=3
else:
    UX=1
    UY=1
T1=time.perf_counter()

for It in range(1001,1001+Iterations):
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(It,1001+Iterations))
    sys.stdout.flush()
    Element_Block=Bezier_Design(E_X,E_Y,UX,UY,Aspect_Ratio,Type,Tesselate,Mesh_Complexity,Overlap)
    Elements,Tri_Elements,Nodes, Element_Plot=UnitCell_To_Metamaterial(E_X,E_Y,UX,UY,Element_Block, Mesh_Complexity, Mirror,Overlap, Tesselate, It,Fillet)
    #np.save('Design_Files/UC_Design_Tess{}_AR{}_{}_Image_{}.npy'.format(UX,Aspect_Ratio,Load_Type,It+1),Element_Plot)
    INP_writer(Elements,Tri_Elements,Nodes,E_X*Mesh_Complexity,E_Y*Mesh_Complexity,UX,UY,Tesselate,Mirror,Strain,AR_Name, Mesh_Complexity, It, Load_Type='C')
    #INP_writer(Elements,Nodes,E_X*Mesh_Complexity,E_Y*Mesh_Complexity,UX,UY,Tesselate,Mirror,Strain,AR_Name, Mesh_Complexity, It, Load_Type='T')
JobScript_Writer(Iterations,Aspect_Ratio)
    
