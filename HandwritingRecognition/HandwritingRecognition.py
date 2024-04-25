"""A script to recognize handwriting using Python.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# Importing necessary librabries
import tensorflow as tf
import tensorflow.data as tfd
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as implt
from IPython.display import clear_output as cls

# Add .python to prevent errors:
# Source: https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2
from tensorflow.python.keras.layers import * 
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 

# Importing the database to train handwriting recognition:


