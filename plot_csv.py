"""
plot_csv.py — Plot a training report CSV as graphs.

Usage:
    python plot_csv.py                        # opens file picker dialog
    python plot_csv.py training_report.csv    # direct path as argument
"""
import sys
import os
import csv
import glob


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(path: str):
    import matplotlib.pyplot as plt

    rows = load_csv(path)
    if not rows:
        print(f"CSV is empty: {path}")
        return

    def col(key, default=0.0):
        return [float(r.get(key) or default) for r in rows]

    def icol(key, default=0):
        return [int(float(r.get(key) or default)) for r in rows]

    gens          = icol("generation")
    best_scores   = col("best_score")
    mean_scores   = col("mean_score")
    median_scores = col("median_score")
    worst_scores  = col("worst_score")
    best_turns    = icol("best_turns")
    median_turns  = col("median_turns")
    worst_turns   = icol("worst_turns")

    has_breakdown = "best_sc_survival" in rows[0]
    has_food      = "best_food_eaten"  in rows[0]

    food_eaten  = icol("best_food_eaten")   if has_food      else []
    sc_survival = col("best_sc_survival")   if has_breakdown else []
    sc_towards  = col("best_sc_towards")    if has_breakdown else []
    sc_against  = col("best_sc_against")    if has_breakdown else []
    sc_ate      = col("best_sc_ate")        if has_breakdown else []
    sc_bomb     = col("best_sc_bomb")       if has_breakdown else []

    plt.style.use("dark_background")

    if has_breakdown:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        ax3 = ax4 = None

    fig.suptitle(f"Bomberman Snake — {os.path.basename(path)}", fontsize=12, fontweight="bold")

    # ── Score overview ────────────────────────────────────────────────────
    ax1.plot(gens, best_scores,   color="#00e676", linewidth=1.5, label="Best")
    ax1.plot(gens, median_scores, color="#ffd740", linewidth=1.5, label="Median")
    ax1.plot(gens, mean_scores,   color="#40c4ff", linewidth=1.0, linestyle="--", label="Mean", alpha=0.8)
    ax1.plot(gens, worst_scores,  color="#ff5252", linewidth=1.0, label="Worst", alpha=0.6)
    ax1.fill_between(gens, best_scores, worst_scores, alpha=0.07, color="white")
    ax1.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax1.set_title("Score (population)", fontsize=10)
    ax1.set_ylabel("Score")
    if not has_breakdown:
        ax1.set_xlabel("Generation")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Turns + food eaten ────────────────────────────────────────────────
    ax2.plot(gens, best_turns,   color="#00e676", linewidth=1.5, label="Best")
    ax2.plot(gens, median_turns, color="#ffd740", linewidth=1.5, label="Median")
    ax2.plot(gens, worst_turns,  color="#ff5252", linewidth=1.0, label="Worst", alpha=0.6)
    ax2.fill_between(gens, best_turns, worst_turns, alpha=0.07, color="white")

    if has_food and any(v > 0 for v in food_eaten):
        ax2r = ax2.twinx()
        ax2r.bar(gens, food_eaten, color="#ff9800", alpha=0.3, width=0.8, label="Food eaten (best)")
        ax2r.set_ylabel("Food eaten", color="#ff9800", fontsize=8)
        ax2r.tick_params(axis="y", labelcolor="#ff9800")
        ax2r.legend(loc="upper right", fontsize=8)

    ax2.set_title("Turns survived" + (" + food eaten" if has_food else ""), fontsize=10)
    ax2.set_ylabel("Turns")
    if not has_breakdown:
        ax2.set_xlabel("Generation")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── Score breakdown: gains ────────────────────────────────────────────
    if has_breakdown:
        ax3.plot(gens, sc_survival, color="#40c4ff", linewidth=1.5, label="Survival")
        ax3.plot(gens, sc_towards,  color="#00e676", linewidth=1.5, label="→ Food (towards)")
        ax3.plot(gens, sc_ate,      color="#ffd740", linewidth=1.5, label="Ate food")
        ax3.axhline(0, color="white", linewidth=0.4, alpha=0.35)
        ax3.set_title("Score breakdown — gains (best agent)", fontsize=10)
        ax3.set_ylabel("Score contribution")
        ax3.set_xlabel("Generation")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, alpha=0.2)

        # ── Score breakdown: penalties ────────────────────────────────────
        ax4.plot(gens, sc_against, color="#ff5252", linewidth=1.5, label="← Food (away)")
        ax4.plot(gens, sc_bomb,    color="#ff9800", linewidth=1.5, label="Bomb penalty")
        ax4.axhline(0, color="white", linewidth=0.4, alpha=0.35)
        ax4.set_title("Score breakdown — penalties (best agent)", fontsize=10)
        ax4.set_ylabel("Score contribution")
        ax4.set_xlabel("Generation")
        ax4.legend(loc="lower left", fontsize=8)
        ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


_HERE = os.path.dirname(os.path.abspath(__file__))


def pick_file_dialog() -> str | None:
    """Open a native file picker. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select training report CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=_HERE,
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def pick_file_terminal() -> str | None:
    """Fallback: list CSV files next to this script and let user pick by number."""
    csvs = sorted(glob.glob(os.path.join(_HERE, "**", "training_report_*.csv"), recursive=True), reverse=True)
    if not csvs:
        print("No training_report_*.csv files found. Pass a path as argument.")
        return None
    print("Available CSV files:")
    for i, f in enumerate(csvs, 1):
        print(f"  [{i}] {f}")
    try:
        choice = int(input("Select number: ").strip())
        return csvs[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            sys.exit(1)
    else:
        path = pick_file_dialog() or pick_file_terminal()

    if path:
        print(f"Plotting: {path}")
        plot(path)


if __name__ == "__main__":
    main()
