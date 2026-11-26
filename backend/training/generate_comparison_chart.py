import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Set style & typography
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Color palette
COLOR_BASE = '#94A3B8'        # Muted Slate for Baseline YOLO11
COLOR_BASE_PAPER = '#0284C7'  # Vibrant Cyan/Blue for Base Paper (Li et al. 2025)
COLOR_PROPOSED = '#10B981'    # Emerald Green for Proposed Framework
COLOR_ACCENT = '#6366F1'      # Indigo Accent
COLOR_WARN = '#F59E0B'        # Amber
COLOR_DANGER = '#EF4444'      # Coral Red
BG_CARD = '#F8FAFC'           # Light Gray Card Background
BORDER_COLOR = '#E2E8F0'

def create_overall_comparison_figure():
    fig = plt.figure(figsize=(19, 12), facecolor='#FFFFFF')
    # Increased top margin to avoid any overlap with header
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 0.95, 1.25], height_ratios=[1.0, 1.15],
                  wspace=0.28, hspace=0.35, top=0.85, bottom=0.06, left=0.05, right=0.96)

    # Supertitle / Header Banner
    fig.text(0.5, 0.960, "Overall Performance Comparison: Proposed Framework vs. Base Paper",
             fontsize=20, weight='bold', ha='center', color='#0F172A')
    fig.text(0.5, 0.925, "Base Paper: Li et al. (Vehicles 2025) YOLO11-AMF  |  Proposed Framework: Adaptive Multi-Stage Framework (AMAVE + SASA + SEDE)",
             fontsize=12, ha='center', color='#475569')

    # =========================================================================
    # PANEL 1: Core Detection & Classification Metrics (Grouped Bar Chart)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_CARD)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_COLOR)

    metrics = ['Precision', 'Recall', 'mAP@50', 'F1-Score']
    baseline_vals = [81.20, 78.50, 82.40, 79.82]
    base_paper_vals = [87.40, 85.10, 88.90, 86.23]
    proposed_vals = [94.10, 91.20, 93.80, 92.63]

    x = np.arange(len(metrics))
    width = 0.25

    rects1 = ax1.bar(x - width, baseline_vals, width, label='Baseline YOLO11', color=COLOR_BASE, edgecolor='#64748B', linewidth=1, alpha=0.9)
    rects2 = ax1.bar(x, base_paper_vals, width, label='Base Paper (YOLO11-AMF)', color=COLOR_BASE_PAPER, edgecolor='#0369A1', linewidth=1, alpha=0.95)
    rects3 = ax1.bar(x + width, proposed_vals, width, label='Proposed Framework (Ours)', color=COLOR_PROPOSED, edgecolor='#047857', linewidth=1.2)

    ax1.set_ylabel('Percentage Score (%)', fontsize=11.5, weight='bold', color='#1E293B')
    ax1.set_title('(A) Primary Detection & Verification Metrics', fontsize=13, weight='bold', color='#0F172A', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10.5, weight='bold', color='#1E293B')
    ax1.set_ylim(65, 103)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, color='#CBD5E1')
    ax1.set_axisbelow(True)

    # Value labels on bars
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.2, color='#475569')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#0369A1')
    for r in rects3:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.8, weight='bold', color='#047857')

    ax1.legend(loc='lower right', framealpha=0.95, fontsize=8.5, edgecolor=BORDER_COLOR)

    # =========================================================================
    # PANEL 2: False Positive Rate & False Alarm Suppression
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_CARD)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_COLOR)

    models_fpr = ['Baseline\nYOLO11', 'Base Paper\n(YOLO11-AMF)', 'Proposed\nFramework']
    fpr_vals = [24.8, 15.3, 0.0]
    bar_colors = [COLOR_DANGER, COLOR_WARN, COLOR_PROPOSED]

    bars = ax2.bar(models_fpr, fpr_vals, width=0.48, color=bar_colors, edgecolor='#334155', linewidth=1)
    ax2.set_ylabel('False Positive Rate (FPR %)', fontsize=11.5, weight='bold', color='#1E293B')
    ax2.set_title('(B) False Alarm Rate Comparison', fontsize=13, weight='bold', color='#0F172A', pad=12)
    ax2.set_ylim(-1, 31)
    ax2.grid(axis='y', linestyle='--', alpha=0.6, color='#CBD5E1')
    ax2.set_axisbelow(True)

    for bar, val in zip(bars, fpr_vals):
        if val > 0:
            ax2.annotate(f"{val:.1f}%\n(Frequent)", xy=(bar.get_x() + bar.get_width() / 2, val),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9, weight='bold', color='#0F172A')
        else:
            ax2.annotate(f"{val:.1f}%\n(0% Event FPR)", xy=(bar.get_x() + bar.get_width() / 2, 0.3),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9.5, weight='bold', color='#047857')

    # Draw arrow showing suppression with non-overlapping position
    ax2.annotate("AMAVE Gating:\nComplete False Alarm\nSuppression",
                 xy=(2, 1.2), xytext=(1.5, 20),
                 arrowprops=dict(facecolor=COLOR_PROPOSED, shrink=0.08, width=1.5, headwidth=6, edgecolor='#047857'),
                 fontsize=8.5, weight='bold', color='#065F46', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec=COLOR_PROPOSED, lw=1))

    # =========================================================================
    # PANEL 3: Radar Chart / Spider Plot Across Operational Dimensions
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    ax3.set_facecolor('#FFFFFF')

    categories = ['Precision', 'Recall', 'mAP@50', 'F1-Score', 'False Alarm\nSuppression', 'Real-Time\nThroughput']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle

    radar_base = [81.2, 78.5, 82.4, 79.8, 75.2, 100.0]
    radar_base_paper = [87.4, 85.1, 88.9, 86.2, 84.7, 92.6]
    radar_proposed = [94.1, 91.2, 93.8, 92.6, 100.0, 85.0]

    radar_base += radar_base[:1]
    radar_base_paper += radar_base_paper[:1]
    radar_proposed += radar_proposed[:1]

    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9, weight='bold', color='#1E293B')
    ax3.tick_params(pad=10)
    ax3.set_rlabel_position(30)
    ax3.set_ylim(60, 105)
    ax3.set_yticks([70, 80, 90, 100])
    ax3.set_yticklabels(['70%', '80%', '90%', '100%'], fontsize=7.5, color='#64748B')
    ax3.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)

    # Plot lines & fill
    ax3.plot(angles, radar_base, linewidth=1.3, linestyle='--', color=COLOR_BASE, label='Baseline YOLO11')
    ax3.plot(angles, radar_base_paper, linewidth=1.8, linestyle='-', color=COLOR_BASE_PAPER, label='Base Paper (YOLO11-AMF)')
    ax3.fill(angles, radar_base_paper, color=COLOR_BASE_PAPER, alpha=0.10)
    ax3.plot(angles, radar_proposed, linewidth=2.4, linestyle='-', color=COLOR_PROPOSED, label='Proposed Framework')
    ax3.fill(angles, radar_proposed, color=COLOR_PROPOSED, alpha=0.22)

    ax3.set_title('(C) Multi-Dimensional Capability Radar', fontsize=12.5, weight='bold', color='#0F172A', pad=16)
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, framealpha=0.95, edgecolor=BORDER_COLOR)

    # =========================================================================
    # PANEL 4: 5-Stage Ablation Progression Curve
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(BG_CARD)
    for spine in ax4.spines.values():
        spine.set_color(BORDER_COLOR)

    configs = ['Config A\n(YOLO11)', 'Config B\n(Base Paper)', 'Config C\n(+AMAVE)', 'Config D\n(+SASA)', 'Config E\n(Proposed Full)']
    f1_progression = [79.82, 86.23, 89.82, 91.62, 92.63]
    p_progression = [81.20, 87.40, 91.50, 93.20, 94.10]
    r_progression = [78.50, 85.10, 88.20, 90.10, 91.20]

    x_line = np.arange(len(configs))
    ax4.plot(x_line, p_progression, marker='o', linewidth=2.2, color=COLOR_PROPOSED, label='Precision (%)', markersize=6.5)
    ax4.plot(x_line, f1_progression, marker='s', linewidth=2.2, color=COLOR_BASE_PAPER, label='F1-Score (%)', markersize=6.5)
    ax4.plot(x_line, r_progression, marker='^', linewidth=2.0, color=COLOR_ACCENT, label='Recall (%)', markersize=6.5)

    for i, val in enumerate(f1_progression):
        ax4.annotate(f"{val:.1f}%", (x_line[i], val), textcoords="offset points", xytext=(0, -14),
                     ha='center', fontsize=8.2, weight='bold', color=COLOR_BASE_PAPER)
    for i, val in enumerate(p_progression):
        ax4.annotate(f"{val:.1f}%", (x_line[i], val), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=8.5, weight='bold', color='#047857')

    ax4.set_xticks(x_line)
    ax4.set_xticklabels(configs, fontsize=8.5, weight='bold', color='#1E293B')
    ax4.set_ylabel('Score (%)', fontsize=11.5, weight='bold', color='#1E293B')
    ax4.set_title('(D) Ablation Gain Across Pipeline Stages', fontsize=13, weight='bold', color='#0F172A', pad=12)
    ax4.set_ylim(75, 98)
    ax4.grid(True, linestyle='--', alpha=0.6, color='#CBD5E1')
    ax4.legend(loc='lower right', fontsize=8.5, framealpha=0.95, edgecolor=BORDER_COLOR)

    # =========================================================================
    # PANEL 5 & 6: Comprehensive Comparison Table & Capability Matrix
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    table_data = [
        ["Metric / Capability", "Baseline YOLO11", "Base Paper (Li et al. 2025)", "Proposed Framework (Ours)", "Gain vs. Base Paper"],
        ["Pipeline Architecture", "Single-Stage YOLO11", "YOLO11 + MLLA + AFPN", "YOLO11-AMF + AMAVE + SASA + SEDE", "Multi-Stage Intelligence"],
        ["Precision", "81.20% (0.812)", "87.40% (0.874)", "94.10% (0.941)", "+6.70% (Superior)"],
        ["Recall", "78.50% (0.785)", "85.10% (0.851)", "91.20% (0.912)", "+6.10% (Superior)"],
        ["mAP @ 0.50", "82.40% (0.824)", "88.90% (0.889)", "93.80% (0.938)", "+4.90% (Superior)"],
        ["F1-Score", "79.82% (0.798)", "86.23% (0.862)", "92.63% (0.926)", "+6.40% (Superior)"],
        ["False Positive Rate (FPR)", "24.8% (Frequent Alarms)", "15.3% (Moderate Alarms)", "0.0% Event / 2.1% Frame", "-13.2% (Suppressed)"],
        ["Throughput / Latency", "28.5 FPS (35.1 ms)", "26.4 FPS (37.9 ms)", "24.2 FPS (41.3 ms)", "Real-Time Capable (>24 FPS)"],
        ["Temporal Verification", "Not Supported", "Not Supported", "Active: AMAVE (5-Frame Window + Flow)", "Novel Contribution 1"],
        ["Severity Assessment", "Not Supported", "Not Supported", "Active: SASA (LOW / MOD / HIGH / CRIT)", "Novel Contribution 2"],
        ["Emergency Decision Routing", "Not Supported", "Not Supported", "Active: SEDE (Multi-Tier Advisory)", "Novel Contribution 3"]
    ]

    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.24, 0.16, 0.18, 0.25, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.48)

    # Style table headers and cells
    for row_idx in range(len(table_data)):
        for col_idx in range(5):
            cell = table[(row_idx, col_idx)]
            cell.set_edgecolor('#CBD5E1')
            cell.set_linewidth(0.8)

            if row_idx == 0:
                cell.set_facecolor('#0F172A')
                cell.set_text_props(weight='bold', color='#FFFFFF', fontsize=9.2)
            else:
                if row_idx % 2 == 1:
                    cell.set_facecolor('#F8FAFC')
                else:
                    cell.set_facecolor('#FFFFFF')

                # Highlight Proposed Framework column
                if col_idx == 3:
                    cell.set_facecolor('#ECFDF5' if row_idx % 2 == 1 else '#F0FDF4')
                    cell.set_text_props(weight='bold', color='#065F46')

                # Highlight Base Paper column
                elif col_idx == 2:
                    cell.set_text_props(color='#0369A1')

                # Highlight Gain column
                elif col_idx == 4:
                    if "+" in table_data[row_idx][col_idx] or "Novel" in table_data[row_idx][col_idx] or "Superior" in table_data[row_idx][col_idx]:
                        cell.set_text_props(weight='bold', color='#047857')
                    elif "-" in table_data[row_idx][col_idx]:
                        cell.set_text_props(weight='bold', color='#0284C7')

                # Left align feature title
                if col_idx == 0:
                    cell.set_text_props(weight='bold', color='#1E293B', ha='left')

    ax5.set_title('(E) System Performance & Quantitative Module Matrix', fontsize=13, weight='bold', color='#0F172A', pad=12)

    # Save to target paths
    save_paths = [
        str(BASE_DIR / "project_output_screenshots" / "overall_performance_comparison.png"),
        str(BASE_DIR / "results" / "plots" / "overall_performance_comparison.png"),
    ]

    for sp in save_paths:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Successfully saved high-resolution figure to: {sp}")

    plt.close()


def create_paper_figure_variant():
    """Generates a compact, 2-column format figure tailored for IEEE/Springer research papers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#FFFFFF', dpi=300)
    plt.subplots_adjust(wspace=0.25, top=0.86, bottom=0.14, left=0.08, right=0.96)

    fig.suptitle("Performance Comparison: Base Paper (YOLO11-AMF) vs. Proposed Multi-Stage Framework",
                 fontsize=14, weight='bold', color='#0F172A', y=0.96)

    # Subplot 1: Bar Chart of Key Metrics
    ax1 = axes[0]
    ax1.set_facecolor(BG_CARD)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_COLOR)

    metrics = ['Precision', 'Recall', 'mAP@50', 'F1-Score']
    base_paper_vals = [87.40, 85.10, 88.90, 86.23]
    proposed_vals = [94.10, 91.20, 93.80, 92.63]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, base_paper_vals, width, label='Base Paper: Li et al. (Vehicles 2025)',
                     color=COLOR_BASE_PAPER, edgecolor='#0369A1', linewidth=1)
    rects2 = ax1.bar(x + width/2, proposed_vals, width, label='Proposed Multi-Stage Framework (Ours)',
                     color=COLOR_PROPOSED, edgecolor='#047857', linewidth=1.2)

    ax1.set_ylabel('Score (%)', fontsize=11, weight='bold', color='#1E293B')
    ax1.set_title('(a) Benchmark Metric Evaluation', fontsize=12, weight='bold', color='#0F172A', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, weight='bold', color='#1E293B')
    ax1.set_ylim(75, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, color='#CBD5E1')
    ax1.set_axisbelow(True)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, weight='bold', color='#0369A1')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.8, weight='bold', color='#047857')

    ax1.legend(loc='lower right', framealpha=0.95, fontsize=8.5, edgecolor=BORDER_COLOR)

    # Subplot 2: Ablation Curve showing incremental gains
    ax2 = axes[1]
    ax2.set_facecolor(BG_CARD)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_COLOR)

    stages = ['Baseline\nYOLO11', 'Base Paper\n(AMF)', '+AMAVE\nVerification', '+SASA\nSeverity', 'Full System\n(+SEDE)']
    p_curve = [81.20, 87.40, 91.50, 93.20, 94.10]
    f1_curve = [79.82, 86.23, 89.82, 91.62, 92.63]
    x_st = np.arange(len(stages))

    ax2.plot(x_st, p_curve, marker='o', linewidth=2.4, color=COLOR_PROPOSED, label='Precision (%)', markersize=7)
    ax2.plot(x_st, f1_curve, marker='s', linewidth=2.4, color=COLOR_BASE_PAPER, label='F1-Score (%)', markersize=7)

    for i in range(len(stages)):
        ax2.annotate(f"{p_curve[i]:.1f}%", (x_st[i], p_curve[i]), textcoords="offset points", xytext=(0, 7),
                     ha='center', fontsize=8.5, weight='bold', color='#047857')
        ax2.annotate(f"{f1_curve[i]:.1f}%", (x_st[i], f1_curve[i]), textcoords="offset points", xytext=(0, -14),
                     ha='center', fontsize=8.2, weight='bold', color='#0369A1')

    ax2.set_xticks(x_st)
    ax2.set_xticklabels(stages, fontsize=8.8, weight='bold', color='#1E293B')
    ax2.set_ylabel('Score (%)', fontsize=11, weight='bold', color='#1E293B')
    ax2.set_title('(b) Stepwise Ablation Gain of Proposed Modules', fontsize=12, weight='bold', color='#0F172A', pad=10)
    ax2.set_ylim(74, 98)
    ax2.grid(True, linestyle='--', alpha=0.6, color='#CBD5E1')
    ax2.legend(loc='lower right', framealpha=0.95, fontsize=8.5, edgecolor=BORDER_COLOR)

    paper_save_paths = [
        str(BASE_DIR / "project_output_screenshots" / "paper_figure_performance_comparison.png"),
        str(BASE_DIR / "results" / "plots" / "paper_figure_performance_comparison.png"),
    ]
    for sp in paper_save_paths:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        plt.savefig(sp, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        print(f"Saved paper figure variant to: {sp}")

    plt.close()

if __name__ == '__main__':
    create_overall_comparison_figure()
    create_paper_figure_variant()
