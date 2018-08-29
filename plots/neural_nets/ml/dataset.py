import numpy

class Dataset:

	def __init__(self):
		pass

	@staticmethod
	def from_geo_tif(f):
		pass

	@staticmethod
	def from_csv(f):
		with open(f) as h:
			data = h.readlines()

			for line in data:
				print (len(line.split(",")))