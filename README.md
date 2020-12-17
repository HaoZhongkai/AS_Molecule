# ASGN

The official implementation of the ASGN model.
Orginal paper: ASGN: An Active Semi-supervised Graph Neural Network for Molecular Property Prediction. KDD'2020 Accepted. 

# Project Structure
+ `base_model`: Containing SchNet and training code for QM9 and OPV datasets. 

+ `rd_learn`: A baseline using random data selection.

+ `geo_learn`: Geometric method of active learning like k_center.

+ `qbc_learn`: Active learning by using query by committee.

+ `utils`: Dataset preparation and utils functions.
+ `baselines`: Active learning baselines from [google's implementation](https://github.com/google/active-learning).

+ `single_model_al`: contains several baseline models and our method ASGN (in file wsl_al.py)

+ `exp`: Experiments loggings.


