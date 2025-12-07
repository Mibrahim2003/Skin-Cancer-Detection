"""
Exploratory Data Analysis (EDA) for HAM10000 Dataset.

This script performs data quality analysis and generates visualizations
for the skin cancer classification project.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metadata(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the HAM10000 metadata CSV file.
    
    Args:
        data_dir: Directory containing the dataset.
        
    Returns:
        DataFrame with metadata.
    """
    metadata_path = Path(data_dir) / "HAM10000_metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "Please run data_ingestion.py first."
        )
    
    df = pd.read_csv(metadata_path)
    print(f"✅ Loaded metadata: {len(df)} records")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing/null values in the DataFrame.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Series with null counts per column.
    """
    print("\n" + "="*50)
    print("MISSING VALUE ANALYSIS")
    print("="*50)
    
    null_counts = df.isnull().sum()
    total_rows = len(df)
    
    for col, count in null_counts.items():
        pct = (count / total_rows) * 100
        status = "⚠️" if count > 0 else "✅"
        print(f"{status} {col}: {count} missing ({pct:.2f}%)")
    
    total_missing = null_counts.sum()
    print(f"\nTotal missing values: {total_missing}")
    
    return null_counts


def analyze_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of diagnosis classes.
    
    Args:
        df: Input DataFrame with 'dx' column.
        
    Returns:
        DataFrame with class counts and percentages.
    """
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Define diagnosis labels for clarity
    diagnosis_labels = {
        'akiec': 'Actinic Keratoses (akiec)',
        'bcc': 'Basal Cell Carcinoma (bcc)',
        'bkl': 'Benign Keratosis (bkl)',
        'df': 'Dermatofibroma (df)',
        'mel': 'Melanoma (mel)',
        'nv': 'Melanocytic Nevi (nv)',
        'vasc': 'Vascular Lesions (vasc)'
    }
    
    # Count classes
    class_counts = df['dx'].value_counts()
    total = len(df)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'diagnosis': class_counts.index,
        'count': class_counts.values,
        'percentage': (class_counts.values / total * 100).round(2)
    })
    
    print("\nClass Distribution Summary:")
    print("-" * 50)
    
    for _, row in summary.iterrows():
        dx = row['diagnosis']
        label = diagnosis_labels.get(dx, dx)
        print(f"  Class '{dx}' ({label.split('(')[0].strip()}): "
              f"{row['count']:,} images ({row['percentage']:.2f}%)")
    
    print("-" * 50)
    print(f"  Total: {total:,} images")
    
    # Identify imbalance
    max_class = summary.iloc[0]
    min_class = summary.iloc[-1]
    imbalance_ratio = max_class['count'] / min_class['count']
    
    print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"   ('{max_class['diagnosis']}' has {imbalance_ratio:.1f}x more samples than '{min_class['diagnosis']}')")
    
    return summary


def create_class_distribution_chart(
    df: pd.DataFrame, 
    output_path: str = "reports/figures/class_distribution.png"
) -> str:
    """
    Create and save a bar chart of class distribution.
    
    Args:
        df: Input DataFrame with 'dx' column.
        output_path: Path to save the figure.
        
    Returns:
        Path to saved figure.
    """
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Create color palette (highlight melanoma in red)
    class_counts = df['dx'].value_counts()
    colors = ['#e74c3c' if dx == 'mel' else '#3498db' for dx in class_counts.index]
    
    # Create bar plot
    ax = sns.countplot(
        data=df, 
        x='dx', 
        order=class_counts.index,
        palette=colors
    )
    
    # Customize plot
    plt.title('HAM10000 Dataset: Class Distribution\n(Skin Cancer Types)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Diagnosis Code', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add value labels on bars
    for i, (dx, count) in enumerate(class_counts.items()):
        ax.text(i, count + 50, f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # Add diagnosis labels as secondary x-axis labels
    diagnosis_names = {
        'nv': 'Melanocytic\nNevi',
        'mel': 'Melanoma\n⚠️',
        'bkl': 'Benign\nKeratosis',
        'bcc': 'Basal Cell\nCarcinoma',
        'akiec': 'Actinic\nKeratoses',
        'vasc': 'Vascular\nLesions',
        'df': 'Dermato-\nfibroma'
    }
    
    ax.set_xticklabels([f"{dx}\n{diagnosis_names.get(dx, '')}" 
                        for dx in class_counts.index], fontsize=9)
    
    # Add legend note
    plt.figtext(0.99, 0.01, 'Red = Malignant (Melanoma)', 
                ha='right', fontsize=9, style='italic', color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Chart saved to: {output_file}")
    return str(output_file)


def analyze_other_features(df: pd.DataFrame):
    """
    Analyze other metadata features like age, sex, localization.
    
    Args:
        df: Input DataFrame.
    """
    print("\n" + "="*50)
    print("ADDITIONAL FEATURE ANALYSIS")
    print("="*50)
    
    # Age distribution
    if 'age' in df.columns:
        print(f"\nAge Statistics:")
        print(f"  Mean: {df['age'].mean():.1f} years")
        print(f"  Median: {df['age'].median():.1f} years")
        print(f"  Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        print(f"  Missing: {df['age'].isnull().sum()} values")
    
    # Sex distribution
    if 'sex' in df.columns:
        print(f"\nSex Distribution:")
        sex_counts = df['sex'].value_counts()
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Localization
    if 'localization' in df.columns:
        print(f"\nLocalization (Top 5):")
        loc_counts = df['localization'].value_counts().head(5)
        for loc, count in loc_counts.items():
            print(f"  {loc}: {count:,} ({count/len(df)*100:.1f}%)")


def run_eda(data_dir: str = "data/raw", output_dir: str = "reports/figures"):
    """
    Run complete EDA pipeline.
    
    Args:
        data_dir: Directory containing the dataset.
        output_dir: Directory to save figures.
    """
    print("="*60)
    print("HAM10000 EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # Load data
    df = load_metadata(data_dir)
    
    # Display basic info
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Check missing values
    check_missing_values(df)
    
    # Analyze class distribution
    class_summary = analyze_class_distribution(df)
    
    # Create visualization
    chart_path = create_class_distribution_chart(
        df, 
        output_path=f"{output_dir}/class_distribution.png"
    )
    
    # Analyze other features
    analyze_other_features(df)
    
    # Print summary for report
    print("\n" + "="*60)
    print("SUMMARY FOR REPORT (Copy to report.md)")
    print("="*60)
    print("""
## Data Quality Observations

### Dataset Overview
- Total samples: {:,}
- Features: {}
- Image format: JPG (600x450 pixels original)

### Class Distribution (Highly Imbalanced)
{}

### Missing Values
{}

### Key Insights
1. **Severe Class Imbalance**: The dataset is highly imbalanced with 'nv' 
   (Melanocytic Nevi) comprising {:.1f}% of all samples.
2. **Melanoma Representation**: Melanoma ('mel') represents only {:.1f}% 
   of the dataset, making it critical to use stratified sampling.
3. **Recommendation**: Use class weights or oversampling techniques during 
   training to handle the imbalance.
""".format(
        len(df),
        list(df.columns),
        class_summary.to_string(index=False),
        df.isnull().sum().to_string(),
        class_summary[class_summary['diagnosis']=='nv']['percentage'].values[0],
        class_summary[class_summary['diagnosis']=='mel']['percentage'].values[0]
    ))
    
    return df, class_summary


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run EDA
    df, summary = run_eda()
    
    print("\n✅ EDA complete!")
