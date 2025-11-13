
# ============================================================================
# TITANIC DATASET - DATA CLEANING & PREPROCESSING
# AI & ML Internship - Task 1
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================================
print("="*80)
print("STEP 1: LOADING AND EXPLORING THE DATASET")
print("="*80)

df = pd.read_csv('Titanic-Dataset.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values Summary:")
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_percent
})
print(missing_df)

print("\nStatistical Summary:")
print(df.describe())

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: HANDLING MISSING VALUES")
print("="*80)

df_clean = df.copy()

# Age: Fill with median (19.87% missing)
print("\n1. Age Column (19.87% missing)")
age_median = df_clean['Age'].median()
print(f"   - Median Age: {age_median}")
df_clean['Age'].fillna(age_median, inplace=True)
print(f"   - Missing values after: {df_clean['Age'].isnull().sum()}")

# Embarked: Fill with mode (0.22% missing)
print("\n2. Embarked Column (0.22% missing)")
embarked_mode = df_clean['Embarked'].mode()[0]
print(f"   - Mode Port: {embarked_mode}")
df_clean['Embarked'].fillna(embarked_mode, inplace=True)
print(f"   - Missing values after: {df_clean['Embarked'].isnull().sum()}")

# Cabin: Drop (77.1% missing - too sparse)
print("\n3. Cabin Column (77.1% missing)")
print("   - Decision: Drop (too many missing values)")
df_clean.drop('Cabin', axis=1, inplace=True)

# Drop non-predictive columns
print("\n4. Dropping Non-Predictive Columns")
print("   - Columns removed: PassengerId, Name, Ticket")
df_clean.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print(f"\nDataset shape after cleaning: {df_clean.shape}")
print(f"Total missing values remaining: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. CONVERT CATEGORICAL FEATURES TO NUMERICAL
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CATEGORICAL ENCODING")
print("="*80)

# Label Encoding for Sex
print("\n1. Sex Column - Label Encoding")
le_sex = LabelEncoder()
df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
encoding_sex = dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
print(f"   - Encoding: {encoding_sex}")

# Label Encoding for Embarked
print("\n2. Embarked Column - Label Encoding")
le_embarked = LabelEncoder()
df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])
encoding_embarked = dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))
print(f"   - Encoding: {encoding_embarked}")

print("\nData after encoding:")
print(df_clean.head())

# ============================================================================
# 4. OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: OUTLIER DETECTION (IQR Method)")
print("="*80)

def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range (IQR) method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
print("\nOutliers detected in each numerical column:")
for col in numerical_cols:
    outliers = detect_outliers_iqr(df_clean, col)
    count = outliers.sum()
    percent = (count / len(df_clean)) * 100
    print(f"   {col}: {count} outliers ({percent:.2f}%)")

# ============================================================================
# 5. NORMALIZE/STANDARDIZE NUMERICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE SCALING (STANDARDIZATION)")
print("="*80)

df_scaled = df_clean.copy()
scaler = StandardScaler()

# Apply standardization to Age and Fare
print("\nApplying StandardScaler to: Age, Fare")
df_scaled[['Age', 'Fare']] = scaler.fit_transform(df_clean[['Age', 'Fare']])

print("\nAfter Standardization:")
print(f"   Age - Mean: {df_scaled['Age'].mean():.6f}, Std: {df_scaled['Age'].std():.6f}")
print(f"   Fare - Mean: {df_scaled['Fare'].mean():.6f}, Std: {df_scaled['Fare'].std():.6f}")

# ============================================================================
# 6. FINAL PREPROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 6: FINAL PREPROCESSED DATASET")
print("="*80)

print(f"\nFinal Dataset Shape: {df_scaled.shape}")
print(f"Total Missing Values: {df_scaled.isnull().sum().sum()}")
print("\nDataset Information:")
print(df_scaled.info())
print("\nFirst 10 rows of preprocessed data:")
print(df_scaled.head(10))
print("\nStatistical Summary of Preprocessed Data:")
print(df_scaled.describe())

# Save preprocessed data
df_scaled.to_csv('titanic_preprocessed.csv', index=False)
print("\n✓ Preprocessed dataset saved to 'titanic_preprocessed.csv'")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: CREATING VISUALIZATIONS")
print("="*80)

# Create figure for outlier visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Outlier Detection - Boxplots', fontsize=16, fontweight='bold')

df_clean_for_plot = df.copy()
df_clean_for_plot['Age'].fillna(age_median, inplace=True)

axes[0, 0].boxplot(df_clean_for_plot['Age'].dropna())
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_ylabel('Age (years)')

axes[0, 1].boxplot(df_clean_for_plot['Fare'].dropna())
axes[0, 1].set_title('Fare Distribution')
axes[0, 1].set_ylabel('Fare (pounds)')

axes[1, 0].boxplot(df_clean_for_plot['SibSp'].dropna())
axes[1, 0].set_title('Siblings/Spouses Distribution')
axes[1, 0].set_ylabel('Count')

axes[1, 1].boxplot(df_clean_for_plot['Parch'].dropna())
axes[1, 1].set_title('Parents/Children Distribution')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
print("\n✓ Outlier detection plot saved to 'outlier_detection_boxplots.png'")

# Create feature importance visualization
fig, ax = plt.subplots(figsize=(10, 6))
features = df_scaled.columns.tolist()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
ax.barh(features, range(len(features)), color=colors)
ax.set_xlabel('Feature Index')
ax.set_title('Final Features After Preprocessing', fontweight='bold', fontsize=14)
ax.set_xticks(range(len(features)))
plt.tight_layout()
plt.savefig('final_features.png', dpi=300, bbox_inches='tight')
print("✓ Final features plot saved to 'final_features.png'")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("\nSummary of Changes:")
print(f"  • Records: {df.shape[0]} → {df_scaled.shape[0]}")
print(f"  • Features: {df.shape[1]} → {df_scaled.shape[1]}")
print(f"  • Missing Values: {df.isnull().sum().sum()} → {df_scaled.isnull().sum().sum()}")
print(f"  • Categorical Features Encoded: Sex, Embarked")
print(f"  • Numerical Features Scaled: Age, Fare")
print("\n✓ All files have been generated successfully!")
