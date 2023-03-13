# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:34:38 2021

@author: nbrow
"""
import numpy as np
import math
def INP_writer(Elements,Tri_Elements,Nodes,E_X,E_Y,UX,UY,Tesselate,Mirror,Strain,Aspect_Ratio,Mesh_Complexity,It,Load_Type):
    All_Elms=np.concatenate((Elements[:,4],Tri_Elements[:,3]))
    if Mirror=='Single':
        Strain_Perc=Strain*UY*E_Y
    else:
        Strain_Perc=Strain*UY*E_Y*2
    with open('INP_Files/UC_Design_NL_{}.inp'.format(It),'w') as f:
        # Writing the Job Info:
        f.write('''*Heading
**Job name: UC_Design_Job_{} Model name: Model-1
** Generated by: Abaqus/CAE 2020
*Preprint, echo=NO, model=NO, history=NO, contact=No
**
'''.format(It))#Incorporate the Parts 
        f.write('''PARTS
**
*Part, name=Body_part
*Node
''')
        for i in range(0,len(Nodes)):
            f.write(' {}, {}, {}\n'.format(int(Nodes[i,2]),Nodes[i,0],Nodes[i,1]))
        f.write('''*Element, type=CPE4RH\n''')
        for i in range(0,len(Elements)):
            f.write(' {}, {}, {}, {}, {}\n'.format(int(Elements[i,4]),int(Elements[i,0]),int(Elements[i,1]),int(Elements[i,2]),int(Elements[i,3])))
        f.write('''*Element, type=CPE3H\n''')
        for i in range(0,len(Tri_Elements)):
            f.write(' {}, {}, {}, {}\n'.format(int(Tri_Elements[i,3]),int(Tri_Elements[i,0]),int(Tri_Elements[i,1]),int(Tri_Elements[i,2])))
        f.write('''*Nset, nset=Set-1, internal\n''')
        for i in range(0,len(Nodes)):
            if i!=len(Nodes)-1:
                f.write('{}, '.format(int(Nodes[i,2])))
            else:
                f.write('{}\n'.format(int(Nodes[i,2])))
            if i%5==0 and i!=0 and i!=len(Elements)-1:
                f.write('\n')
        f.write('*Elset, elset=Set-1, internal\n')
        for i in range(0,len(All_Elms)):
            if i!=len(All_Elms)-1:
                f.write('{}, '.format(int(All_Elms[i])))
            else:
                f.write('{}\n'.format(int(All_Elms[i])))
            if i%5==0 and i!=0 and i!=len(All_Elms)-1:
                f.write('\n')
        Top_Surface=[int(x) for x in Elements[:,4] if x>((E_X*2*UX))*(E_Y*2*UY-1)]
        f.write('\n*Elset, elset=_Body_Surf_S3, internal \n')
        for i in range(0,len(Top_Surface)):
            f.write('{}, '.format(Top_Surface[i]))
            if i%10==0 and i!=0:
                f.write('\n')
        f.write('''\n*Surface, type=ELEMENT, name=Body_Surface
_Body_Surf_S3, S3''')
        f.write('\n** Section: Section-1\n')
        f.write('''*Solid Section, elset=Set-1, material=Material-1
*End Part
**
*Part, name=Rigid
*End Part
**
''')

        f.write('''**
** ASSEMBLY
**
*Assembly, name=Assembly
**
*Instance, name=Part-1-1, part=Body_part
*End Instance
**
*Instance, name=Rigid-1, part=Rigid
 -0.025,   0.05,  0
*Node
1,   0.1,   0, 0
*Nset, nset=Rigid-1-RefPt_, internal
1
*Nset, nset=Rigid_Set
1
*Surface, type=SEGMENTS, name=Rigid_Bottom
START, 0.2, 0
LINE, 0, 0
*Rigid Body, ref node=Rigid-1-RefPt_, analytical surface=Rigid_bottom
*End Instance 
**
*Nset, nset=Bottom_Mid,instance=PART-1-1
{},
'''.format(int((E_X*2*UX)/2)+1))
        El_Len=[]
        Node_Len=[]
        Left_Side=[int(x) for x in Nodes[:,2] if (x-1)%((E_X*UX*2)+1)==0]

        f.write('\n*Nset, nset=Left_Edge, instance=PART-1-1\n')
        for i in range(0,len(Left_Side)):
            f.write('{}, '.format(Left_Side[i]))
            if i%10==0 and i!=0:
                f.write('\n')
        Lower_Side=[int(x) for x in Nodes[:,2] if x<=(E_X*2*UX)+1]
     
        f.write('\n*Nset, nset=Lower_Edge, instance=Part-1-1\n')
        for i in range(0,len(Lower_Side)):
            f.write('{}, '.format(Lower_Side[i]))
            if i%10==0 and i!=0:
                f.write('\n')
        if Mirror=='Single':
            Upper_Side=[int(x) for x in Nodes[:,2] if x>((E_X*2*UX)+1)*(E_Y*UY)]
        else:
            Upper_Side=[int(x) for x in Nodes[:,2] if x>((E_X*2*UX)+1)*(E_Y*2*UY)]

        f.write('\n*Nset, nset=Upper_Edge, instance=Part-1-1\n')
        for i in range(0,len(Upper_Side)):
            f.write('{}, '.format(Upper_Side[i]))
            if i%10==0 and i!=0:
                f.write('\n')
        f.write('''\n*End Assembly
* Amplitude, name=LU
0., 0., 0.2, 1, 0.25, 1, 0.45, 0
**
** MATERIALS
**
*Material, name=Material-1
*Density
 1.235e-08
*Hyperelastic, n=4, reduced polynomial, test data input, moduli=LONG TERM
*Uniaxial Test Data
    -6.6524,  -0.762787
   -6.51544,  -0.757614
   -6.32638,  -0.749937
   -6.16546,  -0.742534
   -6.00813,  -0.733734
   -5.84059,  -0.724323
    -5.6558,  -0.711611
   -5.45537,  -0.699452
   -5.24908,  -0.685099
   -5.07179,  -0.673209
   -4.48864,  -0.655578
   -4.32271,  -0.642971
   -4.17504,  -0.627378
    -4.0309,  -0.615805
   -3.85755,  -0.601809
   -3.42739,  -0.589135
   -3.31762,  -0.576746
   -3.18346,   -0.56293
   -3.07884,  -0.551765
   -2.96383,  -0.539789
   -2.85438,  -0.526966
   -2.70004,  -0.508192
   -2.58371,  -0.493429
   -2.46654,  -0.479662
   -2.39432,  -0.469657
   -2.30569,  -0.456776
   -2.20611,  -0.443658
   -2.09694,  -0.427456
   -2.01391,  -0.414523
   -1.93496,  -0.402103
   -1.85805,  -0.389478
   -1.80398,  -0.380442
   -1.75259,  -0.372067
   -1.67236,  -0.358948
   -1.59426,  -0.345614
   -1.52744,  -0.332864
   -1.46369,  -0.320732
    -1.3977,  -0.308424
   -1.33956,  -0.297612
   -1.28631,  -0.286666
   -1.23536,  -0.276439
   -1.16884,  -0.264358
   -1.11199,  -0.252994
    -1.0619,  -0.243267
   -1.01521,  -0.232538
  -0.961708,  -0.220775
  -0.906875,  -0.209785
  -0.856503,  -0.198668
  -0.812017,  -0.189306
  -0.780352,  -0.182699
  -0.749075,  -0.176047
  -0.714262,  -0.168391
   -0.66903,  -0.158737
  -0.633363,    -0.1529
  -0.582625,  -0.142111
   -0.52564,  -0.129213
  -0.481035,  -0.120189
   -0.44633,  -0.111086
  -0.412977,  -0.103952
  -0.376673,   -0.09574
  -0.347546, -0.0895394
  -0.318436, -0.0822601
  -0.293499, -0.0780785
  -0.268234, -0.0717759
  -0.240088, -0.0643769
  -0.215628, -0.0590615
  -0.187955, -0.0505035
   -0.15223, -0.0429612
  -0.119943, -0.0364559
 -0.0850185, -0.0277238
 -0.0533822, -0.0178197
 -0.0251171, -0.0100572
         0.,         0.
*Hysteresis
2.95,17,6.9,-1
** 
** INTERACTION PROPERTIES
**
*Surface Interaction, name=IntProp-1
1
*Friction
0
*Surface Behavior,no separation, pressure-overclosure=HARD
**
** INTERACTIONS
**
** Interaction: Int-1
*Contact Pair, interaction=IntProp-1
PART-1-1.Body_Surface, Rigid-1.Rigid_Bottom
** ----------------------------------------------------------------''')
        if Load_Type=='C':
            LC_Mag=-1
        else:
            LC_Mag=1
        
        f.write('''\n**
** STEP: Force_Applied_Step
**
*Step, name=Force_Applied_Step, nlgeom=YES, inc=1000
*Dynamic
0.02, 0.45, 4e-06, 0.02
**
** BOUNDARY CONDITIONS
**
** Name: Uniform_Displacement Type: Displacement/Rotation
*Boundary, amplitude=LU
Rigid-1.Rigid_Set, 2,2, -0.01
** Name: Bottom_Edge_BC Type: Displacement/Rotation
*Boundary
Lower_Edge,2,2
** Name:BC-3 Type:Symmetry/Antisymmetry/Encastre
*Boundary
Bottom_Mid, XSYMM
*Boundary
Rigid-1.Rigid_Set, XSYMM
**''')

        f.write('''\n**
** OUTPUT REQUESTS
**
*NODE PRINT, FREQ=1, NSET=Rigid-1.Rigid_Set, SUMMARY=Yes, TOTALS=Yes
U2, RF2
** FIELD OUTPUT: F-Output-1
**
*Output, field, variable=PRESELECT
**
** HISTORY OUTPUT: H-Output-1
**
*Output, history
*Node Output, nset=Rigid-1.Rigid_Set
RF2, U2 
*End Step''')
       

    