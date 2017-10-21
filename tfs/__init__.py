import sys
if sys.version_info[0] >= 3:
	from .TensorFlowInterface import *
	from . import input_data
else:
	from TensorFlowInterface import *
	import input_data
	