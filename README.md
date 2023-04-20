create conda environment by using conda_env.yaml  

data should be organized as data_sample, targets folder is not necessary  

copy config.py.template to config.py and modify it  
then run `python train.py` to start training  

it's recommanded to train on simple card, multiple card training is not tested  

gpu memory usage reference(simple card):
* mixed_precision:
    * batch_size=2: about 12G
    * batch_size=8: about 45G
* full_precision:
    * batch_size=4: about 45G
