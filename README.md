# AL
Active learning for molecular properties prediction
# Project Structure
base_model: containing SchNet and training code for qm9 and opv

rd_learn: A baseline using random data selection

geo_learn: geometric method of active learning like k_center

qbc_learn: active learning by using query by committee 

utils: dataset preparation and utils functions

exp: experiments loggings

# Requirements

pytorch>=1.0.1

rdkit

torchnet>=0.0.4

torchvision>=0.2.2

tensorboardx>=1.6

dgl>=0.3.1


