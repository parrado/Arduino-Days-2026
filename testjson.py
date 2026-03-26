import pandas as pd
from glob import glob
import os
import json
import matplotlib.pyplot as plt
from spectral_features import extract_accel_features
import numpy as np




jsonPath="..//gestos_nano_esp32-export//training"

# Find all files matching the pattern
vertical_files = glob(os.path.join(jsonPath, "vertical*.json"))
sacudir_files = glob(os.path.join(jsonPath, "sacudir*.json"))
giro_files = glob(os.path.join(jsonPath, "giro*.json"))
quieto_files = glob(os.path.join(jsonPath, "quieto*.json"))

# Create a list of DataFrames using a generator expression
# Note: the 'lines=True' parameter might be necessary if your JSON files are in JSON Lines format
# ind_df = (pd.read_json(f, lines=True) for f in all_files)

with open(vertical_files[0], 'r') as f:
    data = json.load(f)
    accel_vertical=data['payload']['values']

with open(sacudir_files[0], 'r') as f:
    data = json.load(f)
    accel_sacudir=data['payload']['values']

with open(giro_files[0], 'r') as f:
    data = json.load(f)
    accel_giro=data['payload']['values']

with open(quieto_files[0], 'r') as f:
    data = json.load(f)
    accel_quieto=data['payload']['values']


feature=np.array(extract_accel_features(accel_vertical))


plt.figure(1)
plt.plot(accel_giro)
plt.title("Acceleration data from giro gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])


plt.figure(2)
plt.plot(accel_sacudir)
plt.title("Acceleration data from sacudir gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])


plt.figure(3)
plt.plot(accel_quieto)
plt.title("Acceleration data from quieto gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])


plt.figure(4)
plt.plot(accel_vertical)
plt.title("Acceleration data from vertical gesture")
plt.xlabel("Index")     
plt.ylabel("Acceleration [m/s^2]")
plt.grid()  # Add grid for better visibility
plt.legend(["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"])

plt.figure(5)
plt.plot(feature[0,:].reshape(9,3))
plt.title("Feature data from vertical gesture")
plt.legend(["X-axis features", "Y-axis features", "Z-axis features"])
plt.grid()  # Add grid for better visibility

plt.show()








