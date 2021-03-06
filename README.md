Machine-learned prediction of the electronic fields in a crystal
================================================================

Written by Ying Shi Teh for the following paper:

Ying Shi Teh, Swarnava Ghosh, and Kaushik Bhattacharya. (2021) Machine-learned prediction of the electronic fields in a crystal, [JMechMat](https://doi.org/10.1016/j.mechmat.2021.104070).

Folder descriptions
==================
* **[00_AbinitRun](00_AbinitRun/)**: 
Data generation and collection of DFT output using the Abinit software.  
* **[01_ML_fields](01_ML_fields/)**: 
Training of PCA/NN models for electronic fields including electron density (DEN), Coulomb potential (VCLMB), band structure energy density (EBAND), and volumetric entropy (ENTR).  
* **[02_ML_XRED](02_ML_XRED/)**:
Training of NN model for atomic positions in fractional (or reduced) coordinates (XRED).  
* **[03_Plots](03_Plots/)**: 
Plots.  
* **[04_EvaluateResults](04_EvaluateResults/)**: 
Evaluate energy values using learned quantities (DEN, EBAND, ENTR, and XRED).  
* **[05_AbinitConvergenceTest](05_AbinitConvergenceTest/)**: 
Convergence test for DFT runs.  
* **[06_ML_ETOT](06_ML_ETOT/)**: 
Training of NN model for total free energy (ETOT). Generate uniaxial stress-strain relationship using learned energy model.
