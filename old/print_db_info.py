#!/usr/bin/env python

import sys
import joblib

db = joblib.load(sys.argv[1])
print(db)
print(db.keys())