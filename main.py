"""
RFE Experiments - Main Entry Point
Runs 3 RFE strategies and compares results
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trainer import run_all_strategies, run_strategy
from config import RFE_STRATEGIES


def main():
    """Main entry point"""
    print("\n" + "="*100)
    print("RFE EXPERIMENTS - FEATURE SELECTION COMPARISON")
    print("="*100)
    print("\nThis experiment will test 3 RFE strategies:")
    print("  1. RFE-only: Pure recursive feature elimination")
    print("  2. Hybrid: SelectKBest (60->40) + RFE (40->25)")
    print("  3. RFE-CV: Cross-validated RFE (automatic feature selection)")
    print("\nModels: XGBoost, LightGBM, Random Forest")
    print("Data: v2 processed data (len/10 window)")
    print("="*100)

    # Run all strategies
    run_all_strategies()

    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*100)
    print("\nCheck the following directories for results:")
    for strategy_key, strategy in RFE_STRATEGIES.items():
        print(f"  {strategy['name']:30s}: {strategy['reports_dir']}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
