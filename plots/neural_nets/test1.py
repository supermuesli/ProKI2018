#!/usr/bin/env python3


import sys
sys.dont_write_bytecode = True

import ml

d = ml.Dataset.from_csv("drugs/drug_consumption.data")