#!/usr/bin/env python

import sys
import struct
from functools import partial
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd


from_file = sys.argv[1]
to_file = sys.argv[2]

import struct

struct_fmt = 'ffff'
struct_len = struct.calcsize(struct_fmt)
struct_unpack = struct.Struct(struct_fmt).unpack_from

with open(from_file, "rb") as f:
  floats = [struct_unpack(chunk) for chunk in iter(partial(f.read, struct_len), b'')]
  

print(len(floats))
points = np.array([np.array(e[:3]) for e in floats])

PyntCloud(pd.DataFrame(data=points,
    columns=["x", "y", "z"])).to_file(to_file)