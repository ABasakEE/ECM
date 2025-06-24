import os
from spyder_kernels.utils.iofuncs import load_dictionary

# Path to your .spydata file
spydata_file = r"RH100,500.spydata"

# Load the contents
namespace, filenames = load_dictionary(spydata_file)

# Now you can access your variables
AST_fit = namespace.get("AST_fit")
circuit_dict = namespace.get("circuit_dict")
DRT_dict = namespace.get("DRT_dict")

# Optional: print to verify
print("Loaded:", list(namespace.keys()))
