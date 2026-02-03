import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit

# --- Define the mathematical function to fit ---
def tanh_func(x, a, b, c, d):
    """
    y = a * tanh(b * (x - c)) + d
    """
    return a * np.tanh(b * (x - c)) + d

class ConductivityMapFitted:
    """
    Creates a conductivity map model by FITTING data from a CSV file
    to a smooth, analytical tanh function.
    """

    def __init__(self, filename):
        self.filename = filename
        self.radius_data_nm = None
        self.conductivity_data = None
        
        # Store the optimal fit parameters
        self.fit_params = None 
        self.ready = False 

        try:
            # 1. Load data
            x_data, y_data = self._load_conductivity_data(self.filename)
            if x_data is None: return

            self.radius_data_nm = x_data
            self.conductivity_data = y_data
            
            # 2. Provide initial guesses (p0) for the fit [a, b, c, d]
            a_guess = (np.max(y_data) - np.min(y_data)) / 2.0
            b_guess = 1.0 # Guess for steepness
            c_guess = np.median(x_data) # Guess for center
            d_guess = (np.max(y_data) + np.min(y_data)) / 2.0 # Guess for vertical shift
            initial_guesses = [a_guess, b_guess, c_guess, d_guess]

            # 3. Perform the curve fit
            # popt = "optimal parameters"
            popt, pcov = curve_fit(
                tanh_func, 
                self.radius_data_nm, 
                self.conductivity_data, 
                p0=initial_guesses
            )
            
            self.fit_params = popt # Save [a, b, c, d]
            self.ready = True
            
            print(f"--- Tanh Fit Model Ready ---")
            print(f"Loaded data from '{filename}'")
            print("Fit Parameters [a, b, c, d]:")
            print(f"  a (amplitude): {popt[0]:.4f}")
            print(f"  b (steepness): {popt[1]:.4f}")
            print(f"  c (center_nm): {popt[2]:.4f}")
            print(f"  d (vertical_shift): {popt[3]:.4f}")
            print("------------------------------")

        except RuntimeError:
            print("Error: Curve fit failed. Could not find optimal parameters.")
            self.ready = False
        except Exception as e:
            print(f"Error during class initialization: {e}")
            self.ready = False

    def _load_conductivity_data(self, filename):
        # (This function is identical to the one in your interpolation class)
        try:
            with open(filename, 'r') as f: lines = f.readlines()
            radius_str = lines[1].strip().split(',')
            radius_nm = np.array(radius_str, dtype=float) * 0.1
            conductivity_str = lines[3].strip().split(',')
            conductivity = np.array(conductivity_str, dtype=float)
            
            if len(radius_nm) != len(conductivity):
                print("Warning: Data length mismatch. Truncating.")
                min_len = min(len(radius_nm), len(conductivity))
                radius_nm = radius_nm[:min_len]
                conductivity = conductivity[:min_len]
            return radius_nm, conductivity
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return None, None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None

    def __call__(self, radius_nm):
        """
        Calculates conductivity using the fitted tanh model.
        """
        if not self.ready:
            print("Error: Model is not ready.")
            return np.nan * np.ones_like(radius_nm)
        
        # Unpack the saved parameters and call the tanh function
        a, b, c, d = self.fit_params
        return tanh_func(radius_nm, a, b, c, d)

# --- Main execution block (Example of using the FITTED class) ---
if __name__ == "__main__":
    
    csv_file = 'ssdna_1M-conductivity-map.csv'
    
    # 1. Create an instance of the FITTED model.
    fitted_map = ConductivityMapFitted(csv_file)
    
    if fitted_map.ready:
        
        # 2. Plot the results
        plt.figure(figsize=(10, 6))
        
        # Plot original noisy data
        plt.scatter(
            fitted_map.radius_data_nm, 
            fitted_map.conductivity_data, 
            label='Original Data', 
            color='blue', 
            alpha=0.5, # Make semi-transparent to see fit
            s=20 # Make points smaller
        )
        
        # Generate a smooth line from the FIT
        x_fit_line = np.linspace(
            np.min(fitted_map.radius_data_nm), 
            np.max(fitted_map.radius_data_nm), 
            200
        )
        
        y_fit_line = fitted_map(x_fit_line) # Use the fitted model
        
        # Unpack params for label
        a, b, c, d = fitted_map.fit_params
        fit_label = f'Tanh Fit: {a:.2f}*tanh({b:.2f}*(x - {c:.2f})) + {d:.2f}'

        plt.plot(
            x_fit_line, 
            y_fit_line, 
            label=fit_label, 
            color='red', 
            linewidth=2.5
        )
        
        plt.title('Conductivity vs. Radius (Tanh Fit Model)')
        plt.xlabel('Radius (nm)')
        plt.ylabel('Normalized Conductivity (S/m**2)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_filename = 'conductivity_tanh_FIT.png'
        plt.savefig(plot_filename)
        print(f"\nPlot saved to '{plot_filename}'")

        # 3. Example Usage
        print("\n--- Model Usage Examples ---")
        radius_1 = 1.0
        print(f"Fitted conductivity at {radius_1} nm: {fitted_map(radius_1):.4f} S/m**2")
