import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_conductivity_data(filename):
    """
    Loads data from the specific CSV format.

    The file is expected to have:
    Line 1: Header (ignored)
    Line 2: Comma-separated radius values
    Line 3: Header (ignored)
    Line 4: Comma-separated conductivity values

    Args:
        filename (str): The path to the CSV file.

    Returns:
        (np.ndarray, np.ndarray): A tuple containing two numpy arrays:
                                  (radius, conductivity)
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Line 2 contains radius data
            radius_str = lines[1].strip().split(',')
            radius = np.array(radius_str, dtype=float)
            
            # Line 4 contains conductivity data
            conductivity_str = lines[3].strip().split(',')
            conductivity = np.array(conductivity_str, dtype=float)
            
            if len(radius) != len(conductivity):
                print(f"Warning: Data length mismatch. {len(radius)} radius values, {len(conductivity)} conductivity values.")
                
            return radius*0.1, conductivity
            
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def tanh_func(x, a, b, c, d):
    """
    Defines a generalized hyperbolic tangent (tanh) function for fitting.
    
    y = a * tanh(b * (x - c)) + d
    
    Args:
        x (float or np.ndarray): Input values (radius).
        a (float): Amplitude (controls the height of the transition).
        b (float): Steepness (controls how sharp the transition is).
        c (float): Horizontal shift (center of the transition).
        d (float): Vertical shift (center of the output).
    
    Returns:
        float or np.ndarray: The calculated y-values.
    """
    return a * np.tanh(b * (x - c)) + d

def ssDNA_1MKCl(filename):
    """
    Loads conductivity data, fits a tanh function, and returns a
    function to calculate conductivity from radius, along with fit data.

    Args:
        filename (str): The path to the CSV data file.

    Returns:
        tuple: A tuple containing:
            - function: A `get_conductivity(radius)` function, or None if fit fails.
            - np.ndarray: The array of optimal parameters [a, b, c, d], or None.
            - np.ndarray: The original x_data (radius), or None.
            - np.ndarray: The original y_data (conductivity), or None.
    """
    # 1. Load the data
    x_data, y_data = load_conductivity_data(filename)
    if x_data is None:
        return None, None, None, None

    # 2. Provide initial guesses (p0) for the fit parameters [a, b, c, d]
    a_guess = (np.max(y_data) - np.min(y_data)) / 2.0
    b_guess = 1.0
    c_guess = np.median(x_data)
    d_guess = (np.max(y_data) + np.min(y_data)) / 2.0
    initial_guesses = [a_guess, b_guess, c_guess, d_guess]

    try:
        # 3. Perform the curve fit
        popt, pcov = curve_fit(tanh_func, x_data, y_data, p0=initial_guesses)
       
        a_fit, b_fit, c_fit, d_fit = popt
        print("--- Fit Successful ---")
        print(f"Optimal Parameters [a, b, c, d]:\n{popt}")
        print("----------------------")

        # 4. Create the returned function using the optimal parameters
        def get_conductivity(radius):
            """
            Calculates conductivity based on the fitted tanh model.
            
            Args:
                radius (float or np.ndarray): The radius/radii to evaluate.
            
            Returns:
                float or np.ndarray: The corresponding fitted conductivity.
            """
            return tanh_func(radius, a_fit, b_fit, c_fit, d_fit)

        # 5. Return the model function, parameters, and original data
        return get_conductivity, popt, x_data, y_data

    except RuntimeError:
        print("Error: Curve fit failed. Could not find optimal parameters.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during fitting: {e}")
        return None, None, None, None

