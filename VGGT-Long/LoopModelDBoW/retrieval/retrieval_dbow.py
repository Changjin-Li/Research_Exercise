import os
import time
from multiprocessing import Process, Queue, Value
import numpy as np
from einops import parse_shape
from collections import OrderedDict

try:
    import dpretrieval
    dpretrieval.DPRetrieval
except:
    print("Couldn't load dpretrieval. It may not be installed.")

import DPRetrieval

class RetrievalDBow:
    def __init__(self, config):
        self.config = config