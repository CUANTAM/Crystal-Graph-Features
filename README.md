# Crystal-Graph-Features
Obtaining geometrical and physical features from crystal ctructure.

The Crystal-Graph-Features jupyter notebook script calculates geometrical and physical attributes derived from the Voronoi tesselation of crystal structure. The atomic-species-aware and directionally biased order parameters are calculated up to the third order of the nearest Voronoi cells for each atom. The script requires the configuration files that are provided in the corresponding folders. The user can provide their own configuration files as well.

The example configurations are all 16 atoms (two conventional cells). The examples are:
1) SiGe (4 monolayers / 4 monolayers) superlattice on Si substrate
2) SiGe (4 monolayers / 4 monolayers) superlattice on Ge substrate
3) SiGe random alloy

Output:

(i) Global attributes:

  1) Configuration effective lattice constants:
  'lat_a', 'lat_b', 'lat_c'
  2) Configuration element wise concentrations:
  'ElementA_conc','ElementB_conc', .....
  3) Atomic positions:
  'x', 'y', 'z'  

(ii) Local attributes:

  1) Unisotropic:
  
  'maximum_packing_efficiency','min_cell_volume', 'max_cell_volume',
  'mean_cell_volume', 'var_cell_volume', 'tot_cell_volume'
  
  (6 total)
  
  2) Biased (have unisotropic and isotropic X, Y, and Z counterparts):
  
  'bond_length', 'var_bond_length','mean_face_area', 'max_face_area',
  'min_face_area', 'var_face_area', 'tot_face_area','coordination',
  'min_electronegativities','max_electronegativities',
  'mean_electronegativities','var_electronegativities',
  'min_ionic_character', 'max_ionic_character', 
  'mean_ionic_character', 'var_ionic_character',
  '1_order', '2_order', '3_order'
  
  (76 total)

Additionally each atom has labels of its 'index' in the configuration and atomic 'symbol'.


/////////////////////////////////////////////////////////////////

Order_Parameters.py

Easy restrat capability from the saved calculated features.

Takes three arguments: number of processors, configuration file path, and output directory. Works with various DFT and MD configuration file formats.
Example:

python Order_Parameters.py 100 ./CONTCAR ./


Calculation time estimate:
Si bulk 40,000 atoms on 1,000 processors ~ 4 hours

