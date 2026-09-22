"""Step 0: collect global market variables (FRED + Yahoo), Feb 28 -> present."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, market  # noqa: E402


def main():
    os.makedirs("data", exist_ok=True)
    levels, changes = market.build_panel()

    levels.to_csv("data/market_levels.csv")
    changes.to_csv("data/market_changes.csv")

    print(f"Window: {changes.index.min().date()} -> {changes.index.max().date()} "
          f"({len(changes)} trading days)")
    print("\n--- levels (last 5) ---")
    print(levels.tail().round(3).to_string())
    print("\n--- daily changes (last 5) ---")
    print(changes.tail().round(4).to_string())
    print("\nNaN counts in changes:")
    print(changes.isna().sum().to_string())


if __name__ == "__main__":
    main()
