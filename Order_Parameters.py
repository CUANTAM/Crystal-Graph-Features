# Import prerequisite packages
# the missing packages can be installed using the command 
#pip install <package name>
#or
#pip3 install <package name>

import pandas as pd
import numpy as np
from pymatgen import Lattice, Structure, Molecule
from pymatgen.analysis import structure_analyzer, local_env
from pymatgen.analysis.chemenv.coordination_environments import voronoi
import os
import sys
#import ase.io
import pymatgen.io.ase
import math
import itertools
import multiprocessing as mp
from operator import itemgetter


# Select an example configuration to get the order parameters
# the desired configuration can be uncommented

#configuration = 'SiGe_Si_substrate'

configuration = 'SiGe_Ge_substrate'

#configuration = 'SiGe_random'


#Get the order parameters
#This may take some time. For multyprocessor machines increase the number of processors for the calculation

#Example:    N_processors = 10


N_processors = 11


#Set the number of output decimals for the calculated features

n_decimals = 3



#Load the configuration

structure = pymatgen.Structure.from_file(configuration+'/CONTCAR')

# Make a supercell to distinguish the atoms from its replicated counterparts (needed only for the small unit cells)



M=[2,2,1]
structure.make_supercell(M)
scale=np.product(M)

lat_a=structure.lattice.a
lat_b=structure.lattice.b
lat_c=structure.lattice.c
lat_a=lat_a/np.round(lat_a/5.475,decimals=0)
lat_b=lat_b/np.round(lat_b/5.475,decimals=0)
lat_c=lat_c/np.round(lat_c/5.475,decimals=0)

sizestructure=len(structure.atomic_numbers)


face_areas_r=[]
neighbors=[]
element_nn_list=[]
face_area_dist=[]
neighbors_xyz=[]
volumes_r=[]

df_all=pd.DataFrame()
df_all_atom=pd.DataFrame()

nn1_old=[]
nn2_old=[]
nn3_old=[]


def calculate_local_properties(dictionary, neighbors, core_atom, face_areas_r):
    dict_core=[dictionary[x] for x in core_atom['symbol']]
    dict_tot=[]
    
    
    for i in range(len(dict_core)):

        element_list_neigh=neighbors['element_nn_list'] #[i]

        areas=face_areas_r #['face_areas_r'] #[i]

    
        values_n=np.array([dictionary[x] for x in element_list_neigh])

        dict_tot.append(np.sum(areas*np.abs(values_n-dict_core[i]))/np.sum(areas))

    features=[np.max(dict_tot), np.min(dict_tot), np.mean(dict_tot), np.sum(np.abs(dict_tot-np.mean(dict_tot)))/len(dict_tot),dict_tot]

    return features


def calculate_ionic_character(dictionary, neighbors, core_atom, face_areas_r):
    
    dict_core=[dictionary[x] for x in core_atom['symbol']]
    dict_tot=[]
    
    
    for i in range(len(dict_core)):

        element_list_neigh=neighbors['element_nn_list']
        areas=face_areas_r #['face_areas_r']

    
        values_n=np.array([dictionary[x] for x in element_list_neigh])

        dict_tot.append(np.sum(areas*np.abs(1-np.exp(-0.25*np.power((values_n-dict_core[i]), 2))))/np.sum(areas))

    features=[np.max(dict_tot), np.min(dict_tot), np.mean(dict_tot), np.sum(np.abs(dict_tot-np.mean(dict_tot)))/len(dict_tot),dict_tot]

    return features
           
            
def get_atomic_env(w):
            global df_all
            global df_all_atom     
            df=pd.DataFrame()
            df_atom=pd.DataFrame()
            cell_dict=local_env.VoronoiNN().get_nn_info(structure,w)
            df['index']=[w]*len(cell_dict)
            df['symbol']=[str(structure._sites[w]).split()[-1]]*len(cell_dict)
            df['neighbors']=([cell_dict[v]['site_index'] for v in range(len(cell_dict))])
            df['element_nn_list']=([cell_dict[v]['site'].species_string for v in range(len(cell_dict))])
            df['volumes_r']=([cell_dict[v]['poly_info']['volume'] for v in range(len(cell_dict))])
            df['face_areas_r']=([cell_dict[v]['poly_info']['area'] for v in range(len(cell_dict))])   
            df['face_area_dist']=([cell_dict[v]['poly_info']['face_dist']*2 for v in range(len(cell_dict))])
            df['x']=[structure.cart_coords[w][0]]*len(cell_dict)
            df['y']=[structure.cart_coords[w][1]]*len(cell_dict)
            df['z']=[structure.cart_coords[w][2]]*len(cell_dict)
            df['neighbors_x']=([cell_dict[v]['site'].coords[0] for v in range(len(cell_dict))])
            df['neighbors_y']=([cell_dict[v]['site'].coords[1] for v in range(len(cell_dict))])
            df['neighbors_z']=([cell_dict[v]['site'].coords[2] for v in range(len(cell_dict))])
            df['dist_x']=abs(df['x']-df['neighbors_x'])
            df['dist_y']=abs(df['y']-df['neighbors_y'])
            df['dist_z']=abs(df['z']-df['neighbors_z'])
            df['sinphi2']=df['dist_y']**2/(df['dist_x']**2+df['dist_y']**2+1E-99)
            df['cosphi2']=df['dist_x']**2/(df['dist_x']**2+df['dist_y']**2+1E-99)
            df['sintheta2']=(df['dist_x']**2+df['dist_y']**2)/(df['dist_x']**2+df['dist_y']**2+df['dist_z']**2+1E-99)
            df['costheta2']=df['dist_z']**2/(df['dist_x']**2+df['dist_y']**2+df['dist_z']**2+1E-99)
            df['face_areas_r_a']=df['face_areas_r']*df['cosphi2']*df['sintheta2']
            df['face_areas_r_b']=df['face_areas_r']*df['sinphi2']*df['sintheta2']
            df['face_areas_r_c']=df['face_areas_r']*df['costheta2']
            df_atom['index']=[w]
            df_atom['symbol']=[str(structure._sites[w]).split()[-1]]
            df_atom['maximum_packing_efficiency']=np.round((4/3*np.pi*(df['face_area_dist'].min()/2)**3)/(df['volumes_r'].sum()),decimals=n_decimals)

            df_atom['bond_length']=np.round(np.dot(df['face_area_dist'],np.transpose(df['face_areas_r']))/df['face_areas_r'].sum(),decimals=n_decimals)
            df_atom['bond_length_a']=np.round(np.dot(df['face_area_dist'],np.transpose(df['face_areas_r_a']))/df['face_areas_r_a'].sum(),decimals=n_decimals)
            df_atom['bond_length_b']=np.round(np.dot(df['face_area_dist'],np.transpose(df['face_areas_r_b']))/df['face_areas_r_b'].sum(),decimals=n_decimals)
            df_atom['bond_length_c']=np.round(np.dot(df['face_area_dist'],np.transpose(df['face_areas_r_c']))/df['face_areas_r_c'].sum(),decimals=n_decimals)
            
            df_atom['var_bond_length']=np.round(np.dot((df['face_area_dist']-df_atom['bond_length'].values)**2, np.transpose(df['face_areas_r']))/(df['face_areas_r'].sum()),decimals=n_decimals)
            df_atom['var_bond_length_a']=np.round(np.dot((df['face_area_dist']-df_atom['bond_length_a'].values)**2, np.transpose(df['face_areas_r_a']))/(df['face_areas_r_a'].sum()),decimals=n_decimals)
            df_atom['var_bond_length_b']=np.round(np.dot((df['face_area_dist']-df_atom['bond_length_b'].values)**2, np.transpose(df['face_areas_r_b']))/(df['face_areas_r_b'].sum()),decimals=n_decimals)
            df_atom['var_bond_length_c']=np.round(np.dot((df['face_area_dist']-df_atom['bond_length_c'].values)**2, np.transpose(df['face_areas_r_c']))/(df['face_areas_r_c'].sum()),decimals=n_decimals)

            df_atom['min_cell_volume']=np.round(df['volumes_r'].min(),decimals=n_decimals)
            df_atom['max_cell_volume']=np.round(df['volumes_r'].max(),decimals=n_decimals)
            df_atom['mean_cell_volume']=np.round(df['volumes_r'].mean(),decimals=n_decimals)
            df_atom['var_cell_volume']=np.round(df['volumes_r'].var(),decimals=n_decimals)
            df_atom['tot_cell_volume']=np.round(df['volumes_r'].sum(),decimals=n_decimals)

            df_atom['mean_face_area']=np.round(df['face_areas_r'].mean() ,decimals=n_decimals)
            df_atom['max_face_area']=np.round(df['face_areas_r'].max(),decimals=n_decimals)
            df_atom['min_face_area']=np.round(df['face_areas_r'].min(),decimals=n_decimals)
            df_atom['var_face_area']=np.round(df['face_areas_r'].var(),decimals=n_decimals)
            df_atom['tot_face_area']=np.round(df['face_areas_r'].sum(),decimals=n_decimals)
            df_atom['mean_face_area_a']=np.round(df['face_areas_r_a'].mean() ,decimals=n_decimals)
            df_atom['max_face_area_a']=np.round(df['face_areas_r_a'].max(),decimals=n_decimals)
            df_atom['min_face_area_a']=np.round(df['face_areas_r_a'].min(),decimals=n_decimals)
            df_atom['var_face_area_a']=np.round(df['face_areas_r_a'].var(),decimals=n_decimals)
            df_atom['tot_face_area_a']=np.round(df['face_areas_r_a'].sum(),decimals=n_decimals)
            df_atom['mean_face_area_b']=np.round(df['face_areas_r_b'].mean() ,decimals=n_decimals)
            df_atom['max_face_area_b']=np.round(df['face_areas_r_b'].max(),decimals=n_decimals)
            df_atom['min_face_area_b']=np.round(df['face_areas_r_b'].min(),decimals=n_decimals)
            df_atom['var_face_area_b']=np.round(df['face_areas_r_b'].var(),decimals=n_decimals)
            df_atom['tot_face_area_b']=np.round(df['face_areas_r_b'].sum(),decimals=n_decimals)
            df_atom['mean_face_area_c']=np.round(df['face_areas_r_c'].mean() ,decimals=n_decimals)
            df_atom['max_face_area_c']=np.round(df['face_areas_r_c'].max(),decimals=n_decimals)
            df_atom['min_face_area_c']=np.round(df['face_areas_r_c'].min(),decimals=n_decimals)
            df_atom['var_face_area_c']=np.round(df['face_areas_r_c'].var(),decimals=n_decimals)
            df_atom['tot_face_area_c']=np.round(df['face_areas_r_c'].sum(),decimals=n_decimals)
            df_atom['coordination']=np.round((df['face_areas_r'].sum())**2/((df['face_areas_r']**2).sum()),decimals=n_decimals)
            df_atom['coordination_a']=np.round((df['face_areas_r_a'].sum())**2/((df['face_areas_r_a']**2).sum()),decimals=n_decimals)
            df_atom['coordination_b']=np.round((df['face_areas_r_b'].sum())**2/((df['face_areas_r_b']**2).sum()),decimals=n_decimals)
            df_atom['coordination_c']=np.round((df['face_areas_r_c'].sum())**2/((df['face_areas_r_c']**2).sum()),decimals=n_decimals)
            electronegativities={"name":'electronegativities',"Al":1.61, "In":1.78, "Ga": 1.81, "O":3.44, "Si" : 1.90, "Ge" : 2.01}
            local_properties=[electronegativities]
            for dictionary in local_properties:
                property_name='electronegativities' #[ k for k,v in locals().items() if v is dictionary][1]    
                features=calculate_local_properties(dictionary, df[['index','element_nn_list']], df_atom[['index','symbol']], df['face_areas_r'])
                df_atom[property_name]=np.round(features[4],decimals=n_decimals)
            features=calculate_ionic_character(electronegativities, df[['index','element_nn_list']],  df_atom[['index','symbol']], df['face_areas_r'])
            df_atom['ionic_character']=np.round(features[4],decimals=n_decimals)
            for dictionary in local_properties:
                property_name='electronegativities_a' #[ k for k,v in locals().items() if v is dictionary][1]    
                features=calculate_local_properties(dictionary, df[['index','element_nn_list']], df_atom[['index','symbol']], df['face_areas_r_a'])
                df_atom[property_name]=np.round(features[4],decimals=n_decimals)
            features=calculate_ionic_character(electronegativities, df[['index','element_nn_list']],  df_atom[['index','symbol']], df['face_areas_r_a'])
            df_atom['ionic_character_a']=np.round(features[4],decimals=n_decimals)
            for dictionary in local_properties:
                property_name='electronegativities_b' #[ k for k,v in locals().items() if v is dictionary][1]    
                features=calculate_local_properties(dictionary, df[['index','element_nn_list']], df_atom[['index','symbol']], df['face_areas_r_b'])
                df_atom[property_name]=np.round(features[4],decimals=n_decimals)
            features=calculate_ionic_character(electronegativities, df[['index','element_nn_list']],  df_atom[['index','symbol']], df['face_areas_r_b'])
            df_atom['ionic_character_b']=np.round(features[4],decimals=n_decimals)
            for dictionary in local_properties:
                property_name='electronegativities_c' #[ k for k,v in locals().items() if v is dictionary][1]    
                features=calculate_local_properties(dictionary, df[['index','element_nn_list']], df_atom[['index','symbol']], df['face_areas_r_c'])
                df_atom[property_name]=np.round(features[4],decimals=n_decimals)
            features=calculate_ionic_character(electronegativities, df[['index','element_nn_list']],  df_atom[['index','symbol']], df['face_areas_r_c'])
            df_atom['ionic_character_c']=np.round(features[4],decimals=n_decimals)            
            df_atom['lat_a']=[lat_a] #*(sizestructure)
            df_atom['lat_b']=[lat_b] #*(sizestructure)
            df_atom['lat_c']=[lat_c] #*(sizestructure)
            df_atom['x']=df['x'].mean()
            df_atom['y']=df['y'].mean()
            df_atom['z']=df['z'].mean()
            df_all=pd.concat([df_all,df])
            df_all_atom=pd.concat([df_all_atom,df_atom])
            return df_all, df_all_atom


indices = np.arange(0,sizestructure,scale)


def slice_iterable(iterable, chunk):
    """
    Slices an iterable into chunks of size n
    :param chunk: the number of items per slice
    :type chunk: int
    :type iterable: collections.Iterable
    :rtype: collections.Generator
    """
    _it = iter(iterable)
    return itertools.takewhile(
        bool, (tuple(itertools.islice(_it, chunk)) for _ in itertools.count(0))
    )


list_of_dict = [dict.fromkeys(indices) for _ in range(12)]




for dictionary in list_of_dict:
    for n in indices:
        dictionary[n]=[]
def worker(enumerated_comps):
    global df_all
    global df_all_atom

    for ind, i in enumerated_comps:
            try:
                if len(df_all[df_all['index']==i])==0:
                    df_all, df_all_atom=get_atomic_env(i)
            except:
                df_all, df_all_atom=get_atomic_env(i)
            

            element_0=df_all[(df_all['index']==i)]['symbol'].drop_duplicates().values
            
            w_tot_1=[]
            w_tot_2=[]
            w_tot_3=[]            
            w_tot_1_a=[]
            w_tot_2_a=[]
            w_tot_3_a=[]
            w_tot_1_b=[]
            w_tot_2_b=[]
            w_tot_3_b=[]
            w_tot_1_c=[]
            w_tot_2_c=[]
            w_tot_3_c=[]            
            for nn1 in df_all[(df_all['index']==i)]['neighbors']: #.drop_duplicates():
                if (i,nn1) not in nn1_old and nn1!=i:
                    nn1_old.append((i,nn1))
                    if len(df_all[df_all['index']==nn1])==0:
                        df_all, df_all_atom=get_atomic_env(nn1)
                    area_tot_1=(df_all[df_all['index']==nn1]['face_areas_r']**1).sum()
                    area_tot_1_a=(df_all[df_all['index']==nn1]['face_areas_r_a']**1).sum()
                    area_tot_1_b=(df_all[df_all['index']==nn1]['face_areas_r_b']**1).sum()
                    area_tot_1_c=(df_all[df_all['index']==nn1]['face_areas_r_c']**1).sum()
                    for area_1, area_1_a, area_1_b, area_1_c, element_1, x, y, z in df_all[(df_all['index']==nn1)&(df_all['neighbors']==i)][['face_areas_r','face_areas_r_a','face_areas_r_b','face_areas_r_c','symbol','dist_x','dist_y','dist_z']].values:
                        if element_1!=element_0:
                            area_1=0
                            area_1_a=0
                            area_1_b=0
                            area_1_c=0                            
                        w_tot_1.append(area_1/area_tot_1)                            
                        w_tot_1_a.append(area_1_a/area_tot_1_a)
                        w_tot_1_b.append(area_1_b/area_tot_1_b)                        
                        w_tot_1_c.append(area_1_c/area_tot_1_c)

                        for nn2 in df_all[(df_all['index']==nn1)]['neighbors']: #.drop_duplicates():
                            if (i,nn1,nn2) not in nn2_old and nn2!=i and nn2!=nn1 and nn1!=i:
                                nn2_old.append((i,nn1,nn2))
                                if len(df_all[df_all['index']==nn2])==0:
                                    df_all, df_all_atom=get_atomic_env(nn2)
                                area_tot_2=(df_all[df_all['index']==nn2 ]['face_areas_r']**1).sum()
                                area_tot_2_a=(df_all[df_all['index']==nn2]['face_areas_r_a']**1).sum()
                                area_tot_2_b=(df_all[df_all['index']==nn2]['face_areas_r_b']**1).sum()
                                area_tot_2_c=(df_all[df_all['index']==nn2]['face_areas_r_c']**1).sum()                                
                                for area_2, area_2_a, area_2_b, area_2_c, element_2, x, y, z in df_all[(df_all['index']==nn2)&(df_all['neighbors']==nn1)][['face_areas_r','face_areas_r_a','face_areas_r_b','face_areas_r_c','symbol','dist_x','dist_y','dist_z']].values:
                                    if element_2!=element_1:
                                        area_2=0
                                        area_2_a=0
                                        area_2_b=0
                                        area_2_c=0                                        
                                    w_tot_2.append(area_2/area_tot_2*area_1/(area_tot_1-area_2))                                        
                                    w_tot_2_a.append(area_2_a/area_tot_2_a*area_1_a/(area_tot_1_a-area_2_a))
                                    w_tot_2_b.append(area_2_b/area_tot_2_b*area_1_b/(area_tot_1_b-area_2_b))
                                    w_tot_2_c.append(area_2_c/area_tot_2_c*area_1_c/(area_tot_1_c-area_2_c))
                                    
                                    for nn3 in df_all[(df_all['index']==nn2)]['neighbors']: #.drop_duplicates():
                                        if (i,nn1,nn2,nn3) not in nn3_old and nn3!=i and nn3!=nn2 and nn3!=nn1 and nn1!=nn2 and nn2!=i and nn1!=i:
                                            nn3_old.append((i,nn1,nn2,nn3))
                                            if len(df_all[df_all['index']==nn3])==0:
                                                df_all, df_all_atom=get_atomic_env(nn3)                                            
                                            area_tot_3=(df_all[df_all['index']==nn3]['face_areas_r']**1).sum() 
                                            area_tot_3_a=(df_all[df_all['index']==nn3]['face_areas_r_a']**1).sum()
                                            area_tot_3_b=(df_all[df_all['index']==nn3]['face_areas_r_b']**1).sum()
                                            area_tot_3_c=(df_all[df_all['index']==nn3]['face_areas_r_c']**1).sum()                                                 
                                            for area_3, area_3_a, area_3_b, area_3_c, element_3, x, y, z in df_all[(df_all['index']==nn3)&(df_all['neighbors']==nn2)][['face_areas_r','face_areas_r_a','face_areas_r_b','face_areas_r_c','symbol','dist_x','dist_y','dist_z']].values:
                                                if element_3!=element_2:
                                                    area_3=0
                                                    area_3_a=0
                                                    area_3_b=0
                                                    area_3_c=0
                                                w_tot_3.append(area_3/area_tot_3*area_2/(area_tot_2-area_3)*area_1/(area_tot_1-area_2))                                                    
                                                w_tot_3_a.append(area_3_a/area_tot_3_a*area_2_a/(area_tot_2_a-area_3_a)*area_1_a/(area_tot_1_a-area_2_a))
                                                w_tot_3_b.append(area_3_b/area_tot_3_b*area_2_b/(area_tot_2_b-area_3_b)*area_1_b/(area_tot_1_b-area_2_b))
                                                w_tot_3_c.append(area_3_c/area_tot_3_c*area_2_c/(area_tot_2_c-area_3_c)*area_1_c/(area_tot_1_c-area_2_b))
            list_of_weights=[w_tot_1,w_tot_2,w_tot_3,w_tot_1_a,w_tot_2_a,w_tot_3_a,w_tot_1_b,w_tot_2_b,w_tot_3_b,w_tot_1_c,w_tot_2_c,w_tot_3_c]                                   


            for k in range(len(list_of_dict)):
                list_of_dict[k][i].append(np.sum(list_of_weights[k]))
                
    df_all_atom.to_csv(configuration+'_atomic_features.csv',index=False)
    return [tuple(k for k in [ind]+list_of_dict)]



all_tuples=indices

comps = tuple(enumerate(all_tuples))

chunksize = int(math.ceil(len(comps)/N_processors))
jobs = tuple(slice_iterable(comps, chunksize))

pool = mp.Pool(processes=N_processors)
work_res = pool.map_async(worker, jobs)

list_of_weights = [[] for _ in range(12)]

for i in indices:
       
    list_of_orders_tmp = [[] for _ in range(12)]

    for result in list(map(itemgetter(1,2,3,4,5,6,7,8,9,10,11,12), sorted(itertools.chain(*work_res.get())))):
    

            
            for k in range(12):
                list_of_orders_tmp[k].append(result[k][i])

    for k in range(12):
        list_of_weights[k].append(sum(np.array(list_of_orders_tmp[k]).sum()))
    


features=['1_order','2_order','3_order','1_order_a','2_order_a','3_order_a','1_order_b','2_order_b','3_order_b','1_order_c','2_order_c','3_order_c']
for k in range(len(features)):
    df_all_atom[features[k]]=np.round(list_of_weights[k],n_decimals)
df_all_atom['index']=indices

df_all_atom_tmp=pd.read_csv(configuration+'_atomic_features.csv')

df_combined=df_all_atom_tmp.merge(df_all_atom,on='index').drop_duplicates('index') #.sort_values(by='z')[['symbol','1_order_a','1_order_b', '1_order_c',  '2_order_a','2_order_b', '2_order_c','3_order_a',  '3_order_b','3_order_c']]



#Save the calculated parameters
#Display calculated order parameters with the atoms ordered alon the Z axis

df_combined.to_csv(configuration+'_combined_features.csv',index=False)
df_combined.drop_duplicates('z').sort_values(by='z')[['symbol','1_order_a','1_order_b', '1_order_c',  '2_order_a','2_order_b', '2_order_c','3_order_a',  '3_order_b','3_order_c']]

df_combined.to_csv(configuration+'_combined_features.csv',index=False)
#df_combined.drop_duplicates('z').sort_values(by='z')[['symbol','1_order_a','1_order_b', '1_order_c',  '2_order_a','2_order_b', '2_order_c','3_order_a',  '3_order_b','3_order_c']]

