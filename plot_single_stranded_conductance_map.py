import numpy as np
import matplotlib.pyplot as plt
from sem_new import ConductivityMapSingleStranded_allatom

# 1. Instantiate the class for the specific concentration
#    Your code is designed to work only with a concentration of 170 mM.
try:
    conductivity_map = ConductivityMapSingleStranded_allatom(concentration=170)
except NotImplementedError:
    print("This class is only implemented for 170 mM KCl.")
    exit()

# 2. Generate a range of distances for the x-axis
#    The data in your class goes from 0 to 2.6 nm.
distances = np.linspace(0, 2.6, 500) # 500 points from 0 to 2.6 nm

# 3. Calculate the conductivity at each distance
#    This "calls" the __call__ method of your object.

conductivities = conductivity_map(np.clip(distances,0,2.6))

# 4. Plot the results using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(distances, conductivities, label='Total Conductivity', color='blue', linewidth=2)

# Add labels and a title for clarity
plt.xlabel("Distance (nm)", fontsize=12)
plt.ylabel("Conductivity (S/m)", fontsize=12)
plt.title("Ionic Conductivity vs. Distance (170 mM KCl)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
