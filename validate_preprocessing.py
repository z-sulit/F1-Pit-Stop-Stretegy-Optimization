import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def validate_preprocessing(laps, original_laps=None):
    print("DATA QUALITY VALIDATION REPORT")

    
    validation_results = {}
    
    # 1. Check for Missing Values
    print("\n1. MISSING VALUES CHECK")
    print("-" * 60)
    # Only check truly critical columns (matching preprocessing)
    critical_columns = ['LapTimeSeconds', 'TyreLife']
    missing_counts = laps[critical_columns].isnull().sum()
    
    if missing_counts.sum() == 0:
        print("No missing values in critical columns")
        validation_results['missing_values'] = 'PASS'
    else:
        print("WARNING: Missing values detected:")
        print(missing_counts[missing_counts > 0])
        validation_results['missing_values'] = 'FAIL'
    
    # 2. Check for Outliers
    print("\n2. OUTLIER CHECK (LapTimeSeconds)")
    print("-" * 60)
    z_scores = np.abs((laps['LapTimeSeconds'] - laps['LapTimeSeconds'].mean()) / laps['LapTimeSeconds'].std())
    outliers = (z_scores > 3).sum()
    
    if outliers == 0:
        print("No outliers detected (|Z| > 3)")
        validation_results['outliers'] = 'PASS'
    else:
        print(f"WARNING: {outliers} potential outliers remain (|Z| > 3)")
        validation_results['outliers'] = 'WARNING'
    
    # 3. Verify Type Conversions
    print("\n3. TYPE CONVERSION CHECK")
    print("-" * 60)
    time_columns = ['LapTimeSeconds', 'Sector1TimeSeconds', 'Sector2TimeSeconds', 'Sector3TimeSeconds']
    all_numeric = all(pd.api.types.is_numeric_dtype(laps[col]) for col in time_columns)
    
    if all_numeric:
        print("All time columns converted to numeric (seconds)")
        validation_results['type_conversion'] = 'PASS'
    else:
        print("ERROR: Some time columns are not numeric")
        validation_results['type_conversion'] = 'FAIL'
    
    # 4. Check Encoding
    print("\n4. CATEGORICAL ENCODING CHECK")
    print("-" * 60)
    compound_cols = [col for col in laps.columns if col.startswith('Compound_')]
    driver_cols = [col for col in laps.columns if col.startswith('Driver_')]
    
    print(f"  Compound encoded features: {len(compound_cols)}")
    print(f"  Driver encoded features: {len(driver_cols)}")
    
    if len(compound_cols) > 0 and len(driver_cols) > 0:
        print("✓ Categorical variables encoded successfully")
        validation_results['encoding'] = 'PASS'
    else:
        print("✗ ERROR: Encoding may have failed")
        validation_results['encoding'] = 'FAIL'
    
    # 5. Check Scaling
    print("\n5. FEATURE SCALING CHECK")
    print("-" * 60)
    scaled_columns = [col for col in laps.columns if col.endswith('_Scaled')]
    
    if len(scaled_columns) > 0:
        print(f"  Scaled features: {len(scaled_columns)}")
        
        # Check if scaled features have mean ≈ 0 and std ≈ 1
        means = laps[scaled_columns].mean()
        stds = laps[scaled_columns].std()
        
        mean_check = all(abs(means) < 0.01)  # Should be close to 0
        std_check = all(abs(stds - 1) < 0.01)  # Should be close to 1
        
        if mean_check and std_check:
            print("✓ Scaling applied correctly (mean ≈ 0, std ≈ 1)")
            validation_results['scaling'] = 'PASS'
        else:
            print("⚠ WARNING: Scaled features may not be properly normalized")
            print(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
            print(f"  Std range: [{stds.min():.4f}, {stds.max():.4f}]")
            validation_results['scaling'] = 'WARNING'
    else:
        print("✗ ERROR: No scaled features found")
        validation_results['scaling'] = 'FAIL'
    
    # 6. Data Distribution Summary
    print("\n6. DATA DISTRIBUTION SUMMARY")
    print("-" * 60)
    print(f"  Total laps: {len(laps)}")
    print(f"  Total features: {len(laps.columns)}")
    print(f"  Unique drivers: {laps['Driver'].nunique()}")
    print(f"  Unique compounds: {laps['Compound'].nunique()}")
    print(f"\n  LapTimeSeconds statistics:")
    print(f"    Mean: {laps['LapTimeSeconds'].mean():.3f}s")
    print(f"    Median: {laps['LapTimeSeconds'].median():.3f}s")
    print(f"    Std Dev: {laps['LapTimeSeconds'].std():.3f}s")
    print(f"    Min: {laps['LapTimeSeconds'].min():.3f}s")
    print(f"    Max: {laps['LapTimeSeconds'].max():.3f}s")
    
    # 7. Before/After Comparison
    if original_laps is not None:
        print("\n7. BEFORE/AFTER COMPARISON")
        print("-" * 60)
        original_count = len(original_laps)
        current_count = len(laps)
        removed = original_count - current_count
        removed_pct = (removed / original_count) * 100
        
        print(f"  Original laps: {original_count}")
        print(f"  After preprocessing: {current_count}")
        print(f"  Removed: {removed} ({removed_pct:.2f}%)")
        
        if removed_pct < 5:
            print("Data retention is good (>95%)")
        elif removed_pct < 10:
            print("Moderate data loss (90-95% retained)")
        else:
            print("WARNING: Significant data loss (>10% removed)")
    
    # Overall Assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    passed = sum(1 for v in validation_results.values() if v == 'PASS')
    total = len(validation_results)
    
    if passed == total:
        print("ALL CHECKS PASSED - Data quality is excellent!")
    elif passed >= total * 0.8:
        print("MOSTLY GOOD - Some warnings, but data is usable")
    else:
        print("ISSUES DETECTED - Review preprocessing carefully")
    
    print(f"  Passed: {passed}/{total} checks")
    print("=" * 60 + "\n")
    
    return validation_results


def plot_preprocessing_diagnostics(laps):
    """
    Creates diagnostic plots to visualize data quality.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Lap Time Distribution
    axes[0, 0].hist(laps['LapTimeSeconds'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Lap Time Distribution (After Preprocessing)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Lap Time (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(laps['LapTimeSeconds'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(laps['LapTimeSeconds'].median(), color='green', linestyle='--', label='Median')
    axes[0, 0].legend()
    
    # 2. Box Plot by Compound
    laps.boxplot(column='LapTimeSeconds', by='Compound', ax=axes[0, 1])
    axes[0, 1].set_title('Lap Time by Tire Compound', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Compound')
    axes[0, 1].set_ylabel('Lap Time (seconds)')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)
    
    # 3. Scaled Features Distribution
    scaled_cols = [col for col in laps.columns if col.endswith('_Scaled')]
    if scaled_cols:
        laps[scaled_cols].boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('Scaled Features Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Scaled Value')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45, ha='right')
    
    # 4. Missing Data Heatmap
    missing_data = laps.isnull().sum().sort_values(ascending=False).head(20)
    if missing_data.sum() > 0:
        missing_data.plot(kind='barh', ax=axes[1, 1])
        axes[1, 1].set_title('Missing Values by Column (Top 20)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Count')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values!', 
                        ha='center', va='center', fontsize=16, fontweight='bold', color='green')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title('Missing Values Check', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# Quick validation function
def quick_check(laps):
    """Quick sanity check - returns True if data looks good."""
    checks = [
        len(laps) > 0,
        'LapTimeSeconds' in laps.columns,
        laps['LapTimeSeconds'].isnull().sum() == 0,
        any(col.startswith('Compound_') for col in laps.columns),
        any(col.endswith('_Scaled') for col in laps.columns)
    ]
    return all(checks)
