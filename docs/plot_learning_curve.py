# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
# ]
# ///

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_logs(log_path: str):
    steps = []
    max_fitness = []
    mean_fitness = []
    max_score = []

    current_step = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Search for the step count (Global Steps)
            step_match = re.search(r"Global Steps\s+(\d+)", line)
            if step_match:
                current_step = int(step_match.group(1))
                continue

            # Search for the row with Fitness values
            if line.startswith("Fitness:") and current_step is not None:
                # Extract the list string from the text: "['-4.55', '-4.57', ...]"
                list_str = line.split("Fitness:")[1].strip()
                try:
                    # Convert string representation of list to an actual list of floats
                    fitness_vals = [float(x) for x in ast.literal_eval(list_str)]

                    steps.append(current_step)
                    max_fitness.append(max(fitness_vals))
                    mean_fitness.append(sum(fitness_vals) / len(fitness_vals))
                except Exception as e:
                    print(f"Error parsing fitness at step {current_step}: {e}")
                continue

            # Additionally - search for Score
            if line.startswith("Score:") and current_step is not None:
                list_str = line.split("Score:")[1].strip()
                try:
                    score_vals = [float(x) for x in ast.literal_eval(list_str)]
                    max_score.append(max(score_vals))
                except Exception:
                    pass

                # Prevent reading the same step logs multiple times
                current_step = None

    return steps, max_fitness, mean_fitness, max_score


def main():
    parser = argparse.ArgumentParser(
        description="Generate learning curve plot from AgileRL logs."
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to the log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="learning-curve.png",
        help="Output image file name",
    )
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Could not find file '{args.log_file}'.")
        return

    print(f"Parsing logs from: {args.log_file} ...")
    steps, max_fitness, mean_fitness, max_score = parse_logs(args.log_file)

    if not steps:
        print("No valid data (Global Steps and Fitness) found in the logs.")
        return

    # Drawing the plot
    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        max_fitness,
        label="Max Fitness (Best Agent)",
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    plt.plot(
        steps,
        mean_fitness,
        label="Mean Fitness (Population)",
        marker="x",
        linewidth=1.5,
        linestyle="--",
        color="#ff7f0e",
    )

    if len(max_score) == len(steps):
        plt.plot(
            steps,
            max_score,
            label="Max Score",
            marker="s",
            linewidth=1,
            linestyle=":",
            color="#2ca02c",
            alpha=0.7,
        )

    plt.title("Learning Curve - AgileRL SlimeVolley", fontsize=14)
    plt.xlabel("Global Steps (Environment Steps)", fontsize=12)
    plt.ylabel("Fitness / Reward", fontsize=12)

    # Format X-axis to display millions (M) neatly
    def millions_formatter(x, pos):
        return f"{x / 1e6:.1f}M"

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save and show
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to file: {args.output}")

    # Display the interactive window
    plt.show()


if __name__ == "__main__":
    main()
