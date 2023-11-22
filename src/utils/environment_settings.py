from easydict import EasyDict as edict

env_settings = edict()
env_settings.DATA = '/home/mamur/TUM/MLMI/data'

# defaults from the paper
env_settings._N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}