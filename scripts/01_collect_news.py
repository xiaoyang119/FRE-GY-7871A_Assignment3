"""Step 1: collect Iran-war news volume (GDELT) and save it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import news  # noqa: E402


def main():
    os.makedirs("data", exist_ok=True)
    vol = news.collect_volume(start="2026-02-28", end="2026-09-20")
    vol.to_csv("data/war_news_volume.csv")
    print(f"volume matrix: {vol.shape} ({vol.index.min()} -> {vol.index.max()})")
    print("\nwar_intensity summary:")
    print(vol["war_intensity"].describe().round(1).to_string())
    print("\ntop-20 war_intensity days:")
    print(vol.sort_values("war_intensity", ascending=False).head(20)[["war_intensity"]].to_string())


if __name__ == "__main__":
    main()
