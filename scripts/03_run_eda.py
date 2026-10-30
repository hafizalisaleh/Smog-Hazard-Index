"""
Script 03: Run Exploratory Data Analysis & Quality Checks (Weeks 1-2).
Generates station data completeness reports, multi-variable correlation matrices,
seasonal trend graphs, and data quality diagnostics.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eda.eda_analysis import EDAAnalyzer


def main():
    print("=" * 70)
    print("STEP 03: EXPLORATORY DATA ANALYSIS (EDA) & QUALITY AUDIT")
    print("=" * 70)

    analyzer = EDAAnalyzer()
    results = analyzer.run_all()

    print("\n[SUCCESS] EDA and quality checks completed:")
    print("  - Station quality report saved : outputs/eda/station_quality_report.csv")
    print("  - Correlation matrix saved     : outputs/eda/variable_correlations.csv")
    print("  - Correlation heatmap saved    : outputs/eda/correlation_heatmap.png")
    print("  - Monthly district trends saved: outputs/eda/monthly_district_pm25_trends.png")


if __name__ == "__main__":
    main()
