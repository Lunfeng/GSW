"""
Generate a simple Jinan housing price trend chart for 2021-2025.

The data is illustrative and aims to provide a clear, readable example of how
prices could evolve over time. Update the `build_price_points` function with
real observations to reflect actual market conditions.
"""
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


PricePoint = tuple[str, float]


def build_price_points() -> list[PricePoint]:
    """Return illustrative monthly average prices for 2021-2025.

    Values represent RMB per square meter and are spaced quarterly to keep the
    chart readable. Replace these with authoritative figures as needed.
    """
    return [
        ("2021-01", 16800),
        ("2021-04", 16950),
        ("2021-07", 17200),
        ("2021-10", 17120),
        ("2022-01", 17280),
        ("2022-04", 17450),
        ("2022-07", 17600),
        ("2022-10", 17730),
        ("2023-01", 17850),
        ("2023-04", 18020),
        ("2023-07", 18240),
        ("2023-10", 18310),
        ("2024-01", 18480),
        ("2024-04", 18560),
        ("2024-07", 18750),
        ("2024-10", 18820),
        ("2025-01", 18990),
        ("2025-04", 19150),
        ("2025-07", 19260),
        ("2025-10", 19380),
    ]


def plot_price_trend(points: list[PricePoint], output_path: Path) -> None:
    dates = [datetime.strptime(point[0], "%Y-%m") for point in points]
    prices = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, prices, marker="o", color="#1f77b4", linewidth=2, label="均价")
    ax.fill_between(dates, prices, color="#1f77b4", alpha=0.12)

    ax.set_title("济南住宅示例价格走势（2021-2025）", fontsize=14, fontweight="bold")
    ax.set_xlabel("月份")
    ax.set_ylabel("均价（元/平方米）")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, linestyle="--", alpha=0.4)

    min_price = min(prices)
    max_price = max(prices)
    min_date = dates[prices.index(min_price)]
    max_date = dates[prices.index(max_price)]

    ax.annotate(
        f"最低点\n{min_price:,.0f}",
        xy=(min_date, min_price),
        xytext=(-30, -25),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#555"),
    )

    ax.annotate(
        f"最高点\n{max_price:,.0f}",
        xy=(max_date, max_price),
        xytext=(20, 25),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#555"),
    )

    ax.legend()
    fig.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成 2021-2025 年济南住宅均价示例图。"
            "可将 build_price_points 中的数据替换为真实观测值。"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jinan_price_trend.png"),
        help="保存图表的路径，默认在当前目录生成 jinan_price_trend.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_points = build_price_points()
    plot_price_trend(price_points, args.output)
    print(f"已生成图表: {args.output.resolve()}")


if __name__ == "__main__":
    main()
