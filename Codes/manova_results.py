import pandas as pd
import glob
import os
from statsmodels.multivariate.manova import MANOVA
import numpy as np

def load_csvs_for_manova(folder_path):
    """
    Loads CSV files that have 'Joint4' and 'Joint5' columns and a filename format like 'pd_0.2_no_dbs.csv'.
    Extracts the severity from the filename and returns a DataFrame with columns:
    ['Severity', 'Joint4', 'Joint5'].
    """
    rows = []
    folder_path = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files"
    pattern = os.path.join(folder_path, "*.csv")
    for file in glob.glob(pattern):
        filename = os.path.basename(file)
        parts = filename.split("_")
        if len(parts) < 2:
            continue
        try:
            # Extract severity from second part, e.g. '0.2' from 'pd_0.2_no_dbs.csv'
            severity_val = float(parts[1])
        except:
            continue
        
        df_temp = pd.read_csv(file)
        if "Joint4" not in df_temp.columns or "Joint5" not in df_temp.columns:
            print(f"Skipping {file}, missing Joint4 or Joint5.")
            continue
        
        for _, row_data in df_temp.iterrows():
            rows.append({
                "Severity": severity_val,
                "Joint4": row_data["Joint4"],
                "Joint5": row_data["Joint5"]
            })
    
    return pd.DataFrame(rows)

def run_manova(df):
    """
    Runs a MANOVA with Joint4 + Joint5 as dependent variables and Severity as a categorical predictor.
    """
    # Convert Severity to categorical
    df["Severity"] = df["Severity"].astype("category")

    # We can use a formula like: 'Joint4 + Joint5 ~ C(Severity)'
    # This tells statsmodels we want to treat 'Severity' as a categorical predictor.
    formula = "Joint4 + Joint5 ~ C(Severity)"
    manova = MANOVA.from_formula(formula, data=df)
    print("=== MANOVA Results ===")
    print(manova.mv_test())

def main():
    folder_path = r"C:\Users\aatis\OneDrive\Desktop\Msc Project\csv files"
    df_all = load_csvs_for_manova(folder_path)
    print(f"Loaded {len(df_all)} rows of data.")
    
    # Run MANOVA
    # This will test if (Joint4, Joint5) differs across Severity groups
    run_manova(df_all)

if __name__ == "__main__":
    main()
