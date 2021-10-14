import argparse
import yaml
from inspect import getmembers, isfunction, isclass
import modeling.structures as structures
import modeling.model as models
from tensorflow import keras

def get_structure(structure_name:str, **kwargs):
    """ Get model structure from name and args.

    Args:
        structure_name: name of structure in modeling.structures

    """
    structs = getmembers(structures, isfunction)
    for s in structs:
        if s[0] == structure_name:
            return s[1](**kwargs)

def get_model(model_name:str, **kwargs):
    """ Get model class from name and args.

    Args:
        model_name: name of structure in modeling.structures

    """
    mods = getmembers(models, isclass)
    for m in mods:
        if m[0] == model_name:
            return m[1](**kwargs)

def get_callbacks(callback_data:dict):
    """ Get callback info from callback_data.

    Args:
        callback_data: config info for callbacks

    """
    callbacks = []
    if callback_data is not False:
        for callback_name in list(callback_data.keys()):
            if callback_name == "EarlyStopping":
                callbacks.append(get_early_stopping(callback_data['EarlyStopping']))
            if callback_name == "ModelCheckpoint":
                callbacks.append(get_model_checkpoint(callback_data['ModelCheckpoint']))
            if callback_name == "TensorBoard":
                callbacks.append(get_tensorboard(callback_data['TensorBoard']))
    return callbacks

def get_early_stopping(callback_config:dict):
    """ Get tf keras EarlyStopping callback.

    Args:
        callback_config: config info to build callback

    """
    return keras.callbacks.EarlyStopping(**callback_config)

def get_model_checkpoint(callback_config:dict):
    """ Get tf keras ModelCheckpoint callback.

    Args:
        callback_config: config info to build callback

    """
    return keras.callbacks.ModelCheckpoint(**callback_config)

def get_tensorboard(callback_config:dict):
    """ Get tf keras Tensorboard callback.

    Args:
        callback_config: config info to build callback

    """
    return keras.callbacks.TensorBoard(**callback_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", help="Path to train run config.")
    args = parser.parse_args()

    with open(args.train_config, "r") as train_config_file:
        train_config_details = yaml.load(train_config_file, Loader=yaml.CLoader)
    
    structure = get_structure(train_config_details['structure']['name'], **train_config_details['structure']['args'])
    model = get_model(train_config_details['model']['name'], **train_config_details['model']['args'])
    callbacks = get_callbacks(train_config_details['callbacks'])
    model.init_model(structure)
    model.train_model(epochs=train_config_details['epochs'], callbacks=callbacks)