# Mapping class (i.e. an OrderedDict with one {key:value})

# Gary Bhumbra

#-------------------------------------------------------------------------------
from collections import OrderedDict
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class mapping (OrderedDict):
  """
  Mapping is an OrderedDict to allow indexed {key:value}, intended for
  convenient pairing of value names with their associated objects e.g.

  {parameter_name: parameter_object}
  {output_name: output_object}
  {input_name: input_object}

  Unfortunately, the possibility of identical names being assigned to
  different objects means that a single mapping alone cannot be used
  for a given values. Therefore DeepNodal uses lists of single mappings
  for model parameters, architectural outputs, and network inputs.
  
  """
  def __init__(self, *args, **kwds):
    OrderedDict.__init__(self, *args, **kwds)

#-------------------------------------------------------------------------------

