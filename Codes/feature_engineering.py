import pandas as pd
import glob
import numpy as np
import os

# =============================
# Feature Engineering Functions
# =============================
def compute_frequency_features(time_data, angle_data):
    """
    Compute frequency-domain features from a time series.
    Returns peak frequency, peak magnitude, and power in the 1-3 Hz band.
    """
    dt = np.mean(np.diff(time_data))
    fs = 1.0 / dt  # Sampling frequency in Hz

    fft_vals = np.fft.fft(angle_data)
    freqs = np.fft.fftfreq(len(angle_data), d=dt)
    mag = np.abs(fft_vals)

    # Only consider non-negative frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    mag_pos = mag[pos_mask]

    # Peak frequency and its magnitude
    peak_idx = np.argmax(mag_pos)
    peak_freq = freqs_pos[peak_idx]
    peak_mag = mag_pos[peak_idx]

    # Compute power in the 1-3 Hz band
    band_mask = (freqs_pos >= 1) & (freqs_pos <= 3)
    band_power = np.sum(mag_pos[band_mask])
    
    return {
        'peak_freq': peak_freq,
        'peak_mag': peak_mag,
        'band_power_1_3Hz': band_power
    }

def compute_stat_features(angle_data):
    """
    Compute basic statistical features: mean, standard deviation, and range.
    """
    mean_val = np.mean(angle_data)
    std_val = np.std(angle_data)
    rng = np.max(angle_data) - np.min(angle_data)
    return {
        'mean_angle': mean_val,
        'std_angle': std_val,
        'range_angle': rng
    }

def compute_kinematic_features(time_data, angle_data):
    """
    Compute kinematic features: velocity and acceleration statistics.
    """
    dt = np.mean(np.diff(time_data))
    velocity = np.diff(angle_data) / dt  # Compute velocity
    acceleration = np.diff(velocity) / dt  if len(velocity) > 1 else np.array([])
    
    vel_mean = np.mean(velocity)
    vel_std = np.std(velocity)
    acc_mean = np.mean(acceleration) if acceleration.size > 0 else 0
    acc_std = np.std(acceleration) if acceleration.size > 0 else 0

    return {
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'acc_mean': acc_mean,
        'acc_std': acc_std
    }

# =============================
# Process Files and Extract Features
# =============================
def extract_features(file_pattern, output_csv):
    """
    Process all CSV files matching the file_pattern.
    Extract features from the "Time" and "Joint4" columns.
    Assumes filenames contain severity as in "pd_0.2_no_dbs.csv" or "pd_0.2_with_dbs.csv".
    Saves a feature table to output_csv.
    """
    feature_rows = []
    
    for file in glob.glob(file_pattern):
        try:
            # Extract severity from filename; assumes format like "pd_0.2_no_dbs.csv"
            severity_str = os.path.basename(file).split('_')[1]
            severity_val = float(severity_str)
        except Exception as e:
            print(f"Error parsing severity from {file}: {e}")
            continue
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        # Ensure required columns exist
        if 'Time' not in df.columns or 'Joint4' not in df.columns:
            print(f"File {file} missing required columns ('Time', 'Joint4').")
            continue
        
        time_data = df['Time'].values
        angle_data = df['Joint4'].values  # You can extend this to include more joints
        
        # Compute features
        freq_feats = compute_frequency_features(time_data, angle_data)
        stat_feats = compute_stat_features(angle_data)
        kin_feats = compute_kinematic_features(time_data, angle_data)
        
        # Combine features into one dictionary
        features = {}
        features.update(freq_feats)
        features.update(stat_feats)
        features.update(kin_feats)
        features["Severity"] = severity_val  # Continuous severity from filename
        
        # Optionally, add file name for reference
        features["SourceFile"] = os.path.basename(file)
        
        feature_rows.append(features)
    
    # Create a DataFrame from the features
    feature_df = pd.DataFrame(feature_rows)
    feature_df.sort_values("Severity", inplace=True)
    
    # Save to CSV
    feature_df.to_csv(output_csv, index=False)
    print(f"Feature table saved to '{output_csv}'.")
    return feature_df

# =============================
# Main Execution
# =============================
# Define base path for CSV files
base_path = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files"

# For PD without DBS:
pattern_no_dbs = os.path.join(base_path, "*no_dbs.csv")
features_pd_no_dbs = extract_features(pattern_no_dbs, "features_pd_no_dbs.csv")

# For PD with DBS:
pattern_with_dbs = os.path.join(base_path, "*with_dbs.csv")
features_pd_with_dbs = extract_features(pattern_with_dbs, "features_pd_with_dbs.csv")
