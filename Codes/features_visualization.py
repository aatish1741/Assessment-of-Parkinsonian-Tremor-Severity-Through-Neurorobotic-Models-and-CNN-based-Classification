import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import os

def visualize_and_analyze(feature_csv, condition_label):
    """
    Loads the feature CSV, generates scatter plots and boxplots of each feature vs Severity,
    and performs one-way ANOVA tests across severity groups.
    
    Parameters:
      feature_csv (str): Path to the feature CSV file.
      condition_label (str): A label for the condition (e.g., "PD without DBS" or "PD with DBS").
      
    Saves:
      - One plot per feature (scatter and boxplot) as PNG files.
      - A CSV file with ANOVA results.
    """
    # Load the feature table
    df = pd.read_csv(feature_csv)
    print(f"Loaded {len(df)} rows from {feature_csv}")

    # Identify feature columns (exclude 'Severity' and any reference columns)
    excluded = ["Severity", "SourceFile"]
    feature_columns = [col for col in df.columns if col not in excluded]
    
    # Create output folder for plots if it doesn't exist
    output_folder = f"plots_{condition_label.replace(' ', '_')}"
    os.makedirs(output_folder, exist_ok=True)
    
    # --- Visualization: Scatter Plots ---
    for feature in feature_columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="Severity", y=feature)
        plt.title(f"{feature} vs Severity ({condition_label})")
        plt.xlabel("Severity")
        plt.ylabel(feature)
        scatter_file = os.path.join(output_folder, f"{feature}_scatter_{condition_label.replace(' ', '_')}.png")
        plt.savefig(scatter_file)
        plt.close()
        print(f"Saved scatter plot: {scatter_file}")
    
    # --- Visualization: Boxplots ---
    for feature in feature_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Severity", y=feature)
        plt.title(f"{feature} Distribution across Severity ({condition_label})")
        plt.xlabel("Severity")
        plt.ylabel(feature)
        boxplot_file = os.path.join(output_folder, f"{feature}_boxplot_{condition_label.replace(' ', '_')}.png")
        plt.savefig(boxplot_file)
        plt.close()
        print(f"Saved boxplot: {boxplot_file}")
    
    # --- Statistical Analysis: One-Way ANOVA ---
    anova_results = {}
    severities = sorted(df["Severity"].unique())
    
    for feature in feature_columns:
        # Create groups for each severity value
        groups = [df[df["Severity"] == sev][feature].dropna().values for sev in severities]
        # Only perform ANOVA if there are at least two groups with more than one sample
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) > 1:
            F, p = f_oneway(*valid_groups)
            anova_results[feature] = {"F_statistic": F, "p_value": p}
        else:
            anova_results[feature] = {"F_statistic": None, "p_value": None}
    
    anova_df = pd.DataFrame(anova_results).T.reset_index().rename(columns={'index': 'Feature'})
    anova_csv = f"anova_results_{condition_label.replace(' ', '_')}.csv"
    anova_df.to_csv(anova_csv, index=False)
    print(f"Saved ANOVA results to {anova_csv}")
    print("\nANOVA Results:")
    print(anova_df)

# ----------------------------
# Main Execution
# ----------------------------

# For PD WITHOUT DBS:
# Use your feature CSV file generated for PD without DBS.
#feature_csv_no_dbs = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files\features_pd_no_dbs.csv"
#visualize_and_analyze(feature_csv_no_dbs, "PD without DBS")

# For PD WITH DBS:
# Uncomment the following lines to process the PD with DBS features.
feature_csv_with_dbs = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files\features_pd_with_dbs.csv"
visualize_and_analyze(feature_csv_with_dbs, "PD with DBS")
