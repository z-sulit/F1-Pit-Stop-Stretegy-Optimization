from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
import numpy as np
import pandas as pd

def preprocess_laps(laps):
    """
    Preprocess F1 lap data: handle missing values, remove outliers, 
    standardize types, encode categories, and scale features.
    """
    
    laps = laps.copy()
    
    # 1. Handle Missing Values (only LapTime and TyreLife)
    print("Step 1: Handling missing values")
    initial_count = len(laps)
    critical_columns = ['LapTime', 'TyreLife']
    laps.dropna(subset=critical_columns, inplace=True)
    print(f"  Removed {initial_count - len(laps)} rows")
    
    # 2. Outlier Removal (preserve pit laps)
    print("\nStep 2: Removing outliers")
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    
    is_pit_lap = laps['PitInTime'].notna() | laps['PitOutTime'].notna()
    non_pit_laps = laps[~is_pit_lap].copy()
    z_scores = stats.zscore(non_pit_laps['LapTimeSeconds'])
    abs_z_scores = np.abs(z_scores)
    
    non_pit_outliers = abs_z_scores >= 3
    outliers_removed = non_pit_outliers.sum()
    
    valid_non_pit_indices = non_pit_laps[~non_pit_outliers].index
    pit_lap_indices = laps[is_pit_lap].index
    valid_indices = valid_non_pit_indices.union(pit_lap_indices)
    
    laps = laps.loc[valid_indices]
    print(f"  Removed {outliers_removed} outliers, preserved {is_pit_lap.sum()} pit laps")
    
    # 3. Type Standardization
    print("\nStep 3: Converting timedelta to seconds")
    time_columns = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in time_columns:
        laps[f'{col}Seconds'] = laps[col].dt.total_seconds()
    print(f"  Converted {len(time_columns)} columns")
    
    # 4. Encoding
    print("\nStep 4: Encoding categorical variables")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(laps[['Compound', 'Driver']])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(['Compound', 'Driver'])
    )
    laps.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    laps = pd.concat([laps, encoded_df], axis=1)
    print(f"  Created {encoded_df.shape[1]} features")
    
    # 5. Scaling (for ML models, raw columns preserved for RL)
    print("\nStep 5: Scaling features")
    scaler = StandardScaler()
    columns_to_scale = [
        'LapTimeSeconds', 
        'Sector1TimeSeconds', 
        'Sector2TimeSeconds', 
        'Sector3TimeSeconds', 
        'TyreLife'
    ]
    scaled_features = scaler.fit_transform(laps[columns_to_scale])
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=[f'{col}_Scaled' for col in columns_to_scale]
    )
    laps = pd.concat([laps, scaled_df], axis=1)
    print(f"  Scaled {len(columns_to_scale)} features")
    
    print("\nPreprocessing complete")
    print(f"Final dataset: {len(laps)} laps, {len(laps.columns)} features")
    print(laps.head())

    return laps

