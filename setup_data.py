#!/usr/bin/env python3
"""
Setup script to generate required data files.

This script generates the data files needed for the project:
- prices_YYYY.csv: Historical SPX/USD price data (one file per year)
- features.csv: Engineered features from prices
- uprx_model.joblib: Trained model (requires features.csv first)

Run this script after cloning the repository to generate all required data files.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from histdata_loader import ensure_histdata_prices
from momentum_lib import bootstrap_env, validate_prices, compute_features, train_model, label_future_returns
import pandas as pd
import joblib


def main():
    """Generate all required data files."""
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Load environment
    env_path = project_root / ".env"
    if env_path.exists():
        bootstrap_env(env_path)
        print("✓ Loaded environment variables")
    else:
        print("⚠ .env file not found, continuing without it")
    
    features_file = data_dir / "features.csv"
    model_file = data_dir / "uprx_model.joblib"
    
    # Step 1: Generate year-based price files (prices_YYYY.csv)
    print("\n" + "=" * 60)
    print("STEP 1: Downloading price data (by year)")
    print("=" * 60)
    print("Files will be saved as prices_YYYY.csv (one per year)")
    print("This may take several minutes on first run...")
    prices = ensure_histdata_prices(
        output_dir=data_dir,
        start_year=2019,
        rebuild=True,
        verbose=True,
    )
    prices = validate_prices(prices)
    print(f"✓ Created year-based price files ({len(prices):,} total rows)")
    
    # Step 2: Generate features.csv
    if features_file.exists():
        print(f"✓ {features_file.name} already exists, skipping feature computation")
        features = pd.read_csv(features_file, parse_dates=[0], index_col=0)
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Computing features")
        print("=" * 60)
        features = compute_features(prices)
        features.to_csv(features_file, index=True)
        print(f"✓ Created {features_file.name} ({features.shape[0]:,} rows, {features.shape[1]} features)")
    
    # Step 3: Generate model
    if model_file.exists():
        print(f"✓ {model_file.name} already exists, skipping model training")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Training model")
        print("=" * 60)
        aligned = features.index.intersection(prices.index)
        X = features.loc[aligned]
        y = label_future_returns(prices.loc[aligned], horizon=5)
        X = X.loc[y.index]
        
        model, report, importances = train_model(X, y)
        joblib.dump(model, model_file)
        print(f"✓ Created {model_file.name}")
        print(f"  Model performance: {report}")
    
    print("\n" + "=" * 60)
    print("✓ All data files ready!")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print("\nYou can now run the notebooks in the notebooks/ directory.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

