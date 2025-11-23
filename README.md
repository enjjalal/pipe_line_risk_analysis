# Pipeline Risk Analysis Tool

This tool performs spatial risk analysis on pipeline infrastructure using sensor data, soil properties, and terrain features to identify high-risk segments that may require maintenance or monitoring.

## Features

- **Risk Score Calculation**: Computes risk scores based on multiple factors
- **Correlation Analysis**: Identifies relationships between different risk factors
- **Spatial Visualization**: Generates maps showing risk distribution
- **Segment Analysis**: Breaks down pipelines into manageable segments for detailed inspection
- **Reporting**: Produces comprehensive reports and visualizations

## Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- GeoPandas and its dependencies (may require additional system libraries on some platforms)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd geo_spatial_h2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # OR
   source .venv/bin/activate  # On Unix/macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis

1. Ensure your input data is in the `integrate` directory
2. Run the analysis script:
   ```bash
   python spatial_risk_analysis.py
   ```

### Output Files

The analysis generates the following outputs in the `visibles` directory:

- `high_risk_segments.geojson`: Geographic data of high-risk pipeline segments
- `pearson_correlation_heatmap.png`: Heatmap of Pearson correlations
- `spearman_correlation_heatmap.png`: Heatmap of Spearman correlations
- `scatter_*.png`: Scatter plots of key variable relationships
- `spatial_risk_map.png`: Map showing risk distribution
- `correlation_report.txt`: Detailed correlation analysis
- `segment_risk_statistics.txt`: Summary statistics of risk scores

## Example Visualizations

### 1. Spatial Risk Map
![Spatial Risk Map](visibles/spatial_risk_map.png)
*Figure 1: Geographic distribution of risk scores across pipeline segments.*

### 2. Correlation Heatmap
![Correlation Heatmap](visibles/pearson_correlation_heatmap.png)
*Figure 2: Pearson correlation matrix showing relationships between different risk factors.*

### 3. Example Scatter Plot
![Example Scatter Plot](visibles/scatter_strain_vs_vibration.png)
*Figure 3: Example scatter plot showing relationship between strain and vibration.*

## Interpreting Results

- **Risk Scores**: Range from 0 (low risk) to 1 (high risk)
- **High-Risk Segments**: Defined as segments with risk scores above the 67th percentile
- **Correlation Values**: Range from -1 (perfect negative) to +1 (perfect positive)

## Customization

You can modify the following parameters in `spatial_risk_analysis.py`:

- `segment_length` in `create_segments_from_sensors()`: Adjust the length of pipeline segments (default: 200m)
- Weights in `calculate_risk_score()`: Adjust the importance of different risk factors
- Visualization parameters: Modify figure sizes, colors, and styles as needed

## Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed
- **Memory Issues**: For large datasets, consider processing in chunks
- **Visualization Errors**: Some plots may fail with certain data types - check the console for specific error messages

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contact

[Your contact information or support email]
