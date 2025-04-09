import pandas as pd
import numpy as np
import os
import glob
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import RandomOverSampler  # For addressing class imbalance

################################################################################
# 1) Data Loading & Labeling
################################################################################
def load_and_label_data(folder_path):
    """
    Loads CSV files in folder_path and assigns labels:
      - 'healthy' => 0.0
      - 'pd_0.x_no_dbs' => float(x)
    Skips 'with_dbs'.
    Returns DataFrame with columns: [Time, Joint4, Joint5, Label, SourceFile].
    """
    rows = []
    pattern = os.path.join(folder_path, "*.csv")
    for file in glob.glob(pattern):
        fname = os.path.basename(file).lower()

        if "with_dbs" in fname:
            continue

        if "healthy" in fname:
            label = 0.0  # healthy
        elif "no_dbs" in fname:
            try:
                # parse severity, e.g. 0.2 from "pd_0.2_no_dbs.csv"
                severity_str = fname.split('_')[1]
                label = float(severity_str)
            except Exception as e:
                print(f"Skipping {file}, error parsing severity: {e}")
                continue
        else:
            print(f"Skipping {file}, cannot determine label.")
            continue

        df_temp = pd.read_csv(file)
        required_cols = ["Time", "Joint4", "Joint5"]
        if not all(col in df_temp.columns for col in required_cols):
            print(f"Skipping {file}, missing {required_cols}.")
            continue

        for _, row_data in df_temp.iterrows():
            rows.append({
                "Time": row_data["Time"],
                "Joint4": row_data["Joint4"],
                "Joint5": row_data["Joint5"],
                "Label": label,
                "SourceFile": os.path.basename(file)
            })

    return pd.DataFrame(rows)

################################################################################
# 2) Feature Engineering
################################################################################
def add_velocity_acceleration(df, group_cols=["SourceFile","Label"]):
    """
    Compute velocity & acceleration for Joint4, Joint5 per file/label group.
    dt=1 assumption for demonstration. Adjust if Time is non-uniform.
    """
    dfs = []
    for _, g in df.groupby(group_cols):
        g = g.sort_values("Time").reset_index(drop=True)

        g["Vel4"] = g["Joint4"].diff().fillna(0)
        g["Vel5"] = g["Joint5"].diff().fillna(0)
        g["Acc4"] = g["Vel4"].diff().fillna(0)
        g["Acc5"] = g["Vel5"].diff().fillna(0)

        dfs.append(g)

    new_df = pd.concat(dfs, ignore_index=True)
    return new_df

def compute_frequency_features(window_data, fs=1.0):
    """
    Example function to compute simple frequency-domain features for a single window.
    'window_data' is shape (timesteps, n_features).
    We can compute, for instance, average power in 1-3 Hz range for each channel.
    
    Returns a 1D array of frequency features for each channel.
    """
    from scipy.fft import rfft, rfftfreq
    n = window_data.shape[0]  # timesteps
    freq_feats = []

    for channel_idx in range(window_data.shape[1]):
        signal = window_data[:, channel_idx]
        # Compute real FFT
        fft_vals = np.abs(rfft(signal))
        freqs = rfftfreq(n, d=1.0/fs)

        # Example: sum power in 1-3 Hz
        band_mask = (freqs >= 8.0) & (freqs <= 20.0)
        band_power = np.sum(fft_vals[band_mask])
        freq_feats.append(band_power)

    return np.array(freq_feats)

def normalize_columns(df, cols):
    """
    Standardize each column in 'cols' to mean=0, std=1.
    """
    for c in cols:
        mean_val = df[c].mean()
        std_val = df[c].std()
        if std_val < 1e-8:
            df[c] = df[c] - mean_val
        else:
            df[c] = (df[c] - mean_val)/std_val
    return df

################################################################################
# 3) Windowing with Overlap + Frequency Features
################################################################################
def create_windows(df, window_size=100, stride=50, base_feature_cols=None, fs=1.0):
    """
    Create overlapping windows of length 'window_size' with stride 'stride'.
    Also compute frequency-domain features for each window and concatenate them.
    
    Returns X (num_samples, window_size, n_base_features + n_freq_features),
            y (num_samples,).
    """
    X_list = []
    y_list = []

    for label, group in df.groupby("Label"):
        group = group.sort_values("Time")
        data_array = group[base_feature_cols].values  # shape: (timesteps, n_base_features)

        i = 0
        while i + window_size <= len(data_array):
            window_data = data_array[i:i+window_size]  # (window_size, n_base_features)

            # Frequency features for each channel
            freq_feats = compute_frequency_features(window_data, fs=fs)  # shape (n_base_features,)

            # Combine raw window + freq feats
            # We'll store the raw window as (window_size, n_base_features),
            # plus a single row of freq features repeated along time dimension or appended differently.
            # A simpler approach: flatten the raw window, then append freq features => 1D
            # But for a CNN, we typically keep the time dimension.
            # We'll do a 2D approach: we add freq features as "extra channels" for each timestep.
            # That means we create a new channel dimension = n_base_features + n_base_features (for freq)?
            # This can get complicated. Alternatively, we can flatten.
            # 
            # We'll do a simpler approach: keep the time dimension for the base signals,
            # then add freq feats as a new "row" at the end => window_size+1 in time dimension.
            # For demonstration, let's do that:
            extended_window = np.vstack([window_data, freq_feats[None, :]])  # shape: (window_size+1, n_base_features)

            X_list.append(extended_window)
            y_list.append(label)
            i += stride

    X = np.array(X_list)  # shape: (num_samples, window_size+1, n_base_features)
    y = np.array(y_list)
    return X, y

################################################################################
# 4) Mapping Severity Floats to Discrete Classes
################################################################################
def map_labels_to_int(y):
    """
    E.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] => [0,1,2,3,4,5]
    """
    unique_vals = sorted(np.unique(y))
    mapping = {val: i for i,val in enumerate(unique_vals)}
    y_int = np.array([mapping[val] for val in y])
    return y_int, mapping

################################################################################
# 5) Build a Deeper CNN
################################################################################
def build_cnn(input_shape, num_classes):
    """
    CNN with extra layers, batch normalization, dropout.
    """
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model

################################################################################
# 6) Oversampling + Class Weight + K-Fold
################################################################################
def stratified_kfold_training_overall(X, y, num_folds=5, epochs=30, batch_size=32):
    """
    - Performs random oversampling to address class imbalance
    - Uses class_weight for further balancing
    - StratifiedKFold cross-validation
    - Prints fold confusion matrices + an overall confusion matrix & classification report
    """
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import RandomOverSampler

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_y_val = []
    all_y_pred = []
    fold_idx = 1

    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weight = {}
    for u, c in zip(unique, counts):
        class_weight[u] = float(total_samples) / (len(unique)*c)

    for train_index, val_index in skf.split(X, y):
        X_train_raw, X_val = X[train_index], X[val_index]
        y_train_raw, y_val = y[train_index], y[val_index]

        # Reshape X_train_raw for oversampling
        # We need (n_samples, n_features) shape to apply RandomOverSampler,
        # but X_train_raw is 3D: (samples, timesteps, channels).
        # We'll flatten them for oversampling, then reshape back.
        nsamples, ntimesteps, nchan = X_train_raw.shape
        X_train_2d = X_train_raw.reshape(nsamples, -1)

        ros = RandomOverSampler(random_state=42)
        X_resampled_2d, y_resampled = ros.fit_resample(X_train_2d, y_train_raw)

        # Reshape back to 3D
        X_resampled = X_resampled_2d.reshape(X_resampled_2d.shape[0], ntimesteps, nchan)

        model = build_cnn(input_shape=(ntimesteps, nchan), num_classes=len(unique))
        model.fit(
            X_resampled, y_resampled,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0
        )

        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFold {fold_idx}: Validation Accuracy = {acc:.4f}")

        y_pred_fold = model.predict(X_val).argmax(axis=1)
        cm_fold = confusion_matrix(y_val, y_pred_fold)
        print(f"Confusion Matrix for Fold {fold_idx}:")
        print(cm_fold)
        print("Explanation: Each row is the actual class, each column is the predicted class.\n")

        all_y_val.extend(y_val)
        all_y_pred.extend(y_pred_fold)

        fold_idx += 1

    # Overall confusion matrix + classification report
    overall_cm = confusion_matrix(all_y_val, all_y_pred)
    overall_report = classification_report(all_y_val, all_y_pred, digits=3)
    print("\n=== Overall Confusion Matrix ===")
    print(overall_cm)
    print("\n=== Overall Classification Report ===")
    print(overall_report)

    # Save overall results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"cnn_improved_results_{timestamp}.txt"
    with open(results_filename, "w") as f:
        f.write("=== Overall CNN Results (Stratified K-Fold) ===\n\n")
        f.write("Applied random oversampling + class weights + freq features.\n\n")
        f.write("Overall Confusion Matrix:\n")
        f.write(str(overall_cm) + "\n\n")
        f.write("Overall Classification Report:\n")
        f.write(overall_report + "\n")
    print(f"Overall results saved to {results_filename}")

################################################################################
# Main
################################################################################
def main():
    folder_path = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files"

    # 1) Load & Label
    df_all = load_and_label_data(folder_path)
    print("Raw data shape:", df_all.shape)

    # 2) Basic Feature Engineering: velocity + acceleration
    df_all = add_velocity_acceleration(df_all)
    base_features = ["Joint4", "Joint5", "Vel4", "Vel5", "Acc4", "Acc5"]
    df_all = normalize_columns(df_all, base_features)

    # 3) Create Overlapping Windows + frequency features
    # We'll keep the sample rate fs=1 for demonstration.
    X, y = create_windows(df_all, window_size=100, stride=50, base_feature_cols=base_features, fs=1.0)
    # Now X has shape (samples, window_size+1, n_base_features)
    # Because we appended freq features as an extra row.

    print("X shape:", X.shape, "y shape:", y.shape)

    # 4) Map severity floats => discrete classes
    y_int, mapping = map_labels_to_int(y)
    print("Label mapping:", mapping)

    # 5) Stratified K-Fold with oversampling + class weights
    stratified_kfold_training_overall(X, y_int, num_folds=5, epochs=30, batch_size=32)

if __name__ == "__main__":
    main()
