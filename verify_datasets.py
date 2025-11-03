"""
Verify Kaggle datasets are properly downloaded and check their structure
Run this BEFORE data_pipeline.py to ensure data is ready
"""

from pathlib import Path
import pandas as pd
import json

def verify_datasets():
    """Verify all required datasets exist and show their structure"""
    
    print("=" * 60)
    print("VERIFYING KAGGLE DATASETS")
    print("=" * 60)
    
    datasets_dir = Path("datasets")
    
    if not datasets_dir.exists():
        print("\n❌ 'datasets/' directory not found!")
        print("   Create it with: mkdir datasets")
        return False
    
    required_datasets = {
        'race': 'RACE Dataset',
        'cefr_texts': 'CEFR Levelled English Texts',
        'iot_learning': 'Language Learning Analysis with IoT',
        'squad': 'Stanford Question Answering Dataset'
    }
    
    all_ok = True
    total_files = 0
    
    for dataset_name, dataset_desc in required_datasets.items():
        dataset_path = datasets_dir / dataset_name
        
        print(f"\n📁 Checking: {dataset_desc}")
        print(f"   Path: {dataset_path}")
        
        if not dataset_path.exists():
            print(f"   ❌ NOT FOUND - Please download this dataset")
            all_ok = False
            continue
        
        # Count files
        csv_files = list(dataset_path.glob("**/*.csv"))
        json_files = list(dataset_path.glob("**/*.json"))
        all_files = list(dataset_path.glob("**/*.*"))
        
        print(f"   ✅ Found: {len(all_files)} files")
        print(f"      - CSV files: {len(csv_files)}")
        print(f"      - JSON files: {len(json_files)}")
        
        total_files += len(all_files)
        
        # Show sample file structure
        if csv_files:
            print(f"\n   📄 Sample CSV: {csv_files[0].name}")
            try:
                df = pd.read_csv(csv_files[0], nrows=0)
                print(f"      Columns: {list(df.columns)}")
            except Exception as e:
                print(f"      ⚠️  Error reading: {e}")
        
        if json_files and dataset_name != 'squad':  # Don't try to load large SQuAD files
            print(f"\n   📄 Sample JSON: {json_files[0].name}")
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"      Keys: {list(data.keys())[:5]}")
                    elif isinstance(data, list):
                        print(f"      Array with {len(data)} items")
            except Exception as e:
                print(f"      ⚠️  Error reading: {e}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL DATASETS FOUND!")
        print(f"   Total files: {total_files}")
        print("\n🚀 Ready to run: python data_pipeline.py")
    else:
        print("❌ SOME DATASETS MISSING")
        print("\n📥 Download missing datasets:")
        print("   1. Go to Kaggle.com")
        print("   2. Search for each dataset:")
        for dataset_name, dataset_desc in required_datasets.items():
            dataset_path = datasets_dir / dataset_name
            if not dataset_path.exists():
                print(f"      - {dataset_desc}")
        print("   3. Download and extract to datasets/ folder")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    verify_datasets()
