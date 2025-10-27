"""
Quick script to check data distribution
"""
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python check_data.py <parquet_file>")
    sys.exit(1)

file_path = sys.argv[1]

print(f"\n📊 Checking {file_path}...\n")

df = pd.read_parquet(file_path)

print(f"Total frames: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")

if 'label' in df.columns:
    print(f"\n🏷️ Label Distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"\nPercentages:")
    print(df['label'].value_counts(normalize=True).sort_index() * 100)

    print(f"\n📈 Summary:")
    print(f"  Normal (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.2f}%)")
    print(f"  Attack (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.2f}%)")
else:
    print("\n⚠️ No 'label' column found!")

print(f"\n✅ Check complete!")
