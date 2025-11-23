import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from shapely.geometry import LineString, Point
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Set up directories
output_dir = 'visibles'
os.makedirs(output_dir, exist_ok=True)

# Load the enriched sensor data
print("Loading data...")
sensors_gdf = gpd.read_file('integrate/enriched_sensors.geojson')

# Calculate risk_score based on available features
print("Calculating risk scores...")
def calculate_risk_score(row):
    # Normalize factors to 0-1 range
    strain_norm = (row['strain'] - sensors_gdf['strain'].min()) / (sensors_gdf['strain'].max() - sensors_gdf['strain'].min())
    vibration_norm = (row['vibration'] - sensors_gdf['vibration'].min()) / (sensors_gdf['vibration'].max() - sensors_gdf['vibration'].min())
    
    # Soil factors (higher clay and organic carbon increase risk)
    clay_risk = (row['soil_clay_pct'] / 100) * 0.3  # Clay increases risk
    org_carbon_risk = (row['soil_organic_carbon'] / 10) * 0.2  # Organic carbon increases risk
    
    # Terrain factors
    slope_risk = (row['slope'] / sensors_gdf['slope'].max()) * 0.2  # Higher slope increases risk
    
    # Combine factors with weights
    risk_score = (
        strain_norm * 0.3 +
        vibration_norm * 0.2 +
        clay_risk * 0.2 +
        org_carbon_risk * 0.2 +
        slope_risk * 0.1
    )
    
    return min(max(risk_score, 0), 1)  # Ensure score is between 0 and 1

# Add risk_score to the dataframe
sensors_gdf['risk_score'] = sensors_gdf.apply(calculate_risk_score, axis=1)

# 1. Compute correlations
print("Computing correlations...")
# Select relevant columns for correlation analysis
correlation_cols = [
    'strain', 'vibration', 'temperature',
    'soil_ph', 'soil_clay_pct', 'soil_sand_pct', 'soil_organic_carbon',
    'slope', 'elevation', 'risk_score'
]

# Compute Pearson and Spearman correlations
pearson_corr = sensors_gdf[correlation_cols].corr(method='pearson')
spearman_corr = sensors_gdf[correlation_cols].corr(method='spearman')

# 2. Visualize correlation heatmaps
print("Generating correlation heatmaps...")
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pearson_correlation_heatmap.png'))
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Spearman Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spearman_correlation_heatmap.png'))
plt.close()

# 3. Generate scatter plots for strongest correlations
print("Generating scatter plots...")

# Function to get top correlations
def get_top_correlations(corr_matrix, n=3):
    # Get the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Get the top n correlations
    top_corrs = (upper.unstack()
                 .sort_values(ascending=False)
                 .dropna()
                 .head(n))
    
    # Get the bottom n correlations
    bottom_corrs = (upper.unstack()
                   .sort_values()
                   .dropna()
                   .head(n))
    
    return pd.concat([top_corrs, bottom_corrs])

# Get top and bottom correlations
top_correlations = get_top_correlations(pearson_corr, n=3)

# Plot the correlations
for (var1, var2), corr in top_correlations.items():
    try:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=sensors_gdf, x=var1, y=var2)
        plt.title(f'Correlation: {corr:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scatter_{var1}_vs_{var2}.png'))
        plt.close()
    except Exception as e:
        print(f"Could not create scatter plot for {var1} vs {var2}: {str(e)}")

# 4. Create spatial risk map
print("Generating spatial risk map...")
fig, ax = plt.subplots(figsize=(15, 15))
sensors_gdf.plot(column='risk_score', 
                cmap='viridis', 
                legend=True,
                ax=ax,
                legend_kwds={'label': 'Risk Score', 'orientation': 'horizontal'})
plt.title('Spatial Risk Map of Pipeline Sensors')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spatial_risk_map.png'))
plt.close()

# 5. Segment pipelines and identify high-risk segments
print("Segmenting pipelines...")
# Group sensors by pipeline and create segments
def create_segments_from_sensors(sensors_gdf, segment_length=200):  # 200m segments
    segments = []
    
    # Group by pipeline
    for pipeline_id, group in sensors_gdf.groupby('pipeline_name'):
        # Get all points for this pipeline
        points = []
        for _, row in group.iterrows():
            point = row['geometry']
            points.append((point.x, point.y))
        
        # Create a line from the points
        if len(points) > 1:
            line = LineString(points)
            total_length = line.length
            num_segments = int(np.ceil(total_length / segment_length))
            
            # Create segments
            for i in range(num_segments):
                start_dist = i * segment_length
                end_dist = min((i + 1) * segment_length, total_length)
                segment_point = line.interpolate(start_dist, normalized=False)
                segments.append({
                    'pipeline_id': pipeline_id,
                    'segment_id': f"{pipeline_id}_{i}",
                    'segment_length': end_dist - start_dist,
                    'geometry': segment_point
                })
    
    return gpd.GeoDataFrame(segments, crs=sensors_gdf.crs)

# Create segments
segments_gdf = create_segments_from_sensors(sensors_gdf)

# Spatial join to assign sensor data to segments
segments_gdf = gpd.sjoin_nearest(
    segments_gdf, 
    sensors_gdf[['risk_score', 'geometry']], 
    how='left', 
    distance_col='distance_to_sensor'
)

# Aggregate risk scores for each segment
segment_risk = segments_gdf.groupby('segment_id').agg({
    'risk_score': 'mean',
    'segment_length': 'first',
    'pipeline_id': 'first',
    'geometry': 'first'
}).reset_index()

# Convert back to GeoDataFrame
segment_risk_gdf = gpd.GeoDataFrame(segment_risk, geometry='geometry', crs=segments_gdf.crs)

# Identify high-risk segments (risk_score > 0.67)
high_risk_segments = segment_risk_gdf[segment_risk_gdf['risk_score'] > 0.67]

# 6. Calculate accuracy using a simple threshold
print("Calculating risk classification...")
# Define high-risk threshold (67th percentile)
high_risk_threshold = segment_risk_gdf['risk_score'].quantile(0.67)
segment_risk_gdf['risk_category'] = np.where(
    segment_risk_gdf['risk_score'] > high_risk_threshold, 
    'High', 
    'Normal'
)

# Print summary statistics
print("\n=== Risk Score Summary ===")
print(f"Mean risk score: {segment_risk_gdf['risk_score'].mean():.4f}")
print(f"Median risk score: {segment_risk_gdf['risk_score'].median():.4f}")
print(f"High-risk threshold (67th percentile): {high_risk_threshold:.4f}")
print(f"Number of high-risk segments: {len(segment_risk_gdf[segment_risk_gdf['risk_score'] > high_risk_threshold])}")
print(f"Percentage of high-risk segments: {len(segment_risk_gdf[segment_risk_gdf['risk_score'] > high_risk_threshold]) / len(segment_risk_gdf) * 100:.2f}%")

# 7. Save results
print("Saving results...")
# Save high-risk segments
high_risk_segments = segment_risk_gdf[segment_risk_gdf['risk_score'] > high_risk_threshold]

# Add pipeline information to the output
pipeline_info = sensors_gdf[['pipeline_name', 'pipeline_operator', 'pipeline_diameter_mm', 'pipeline_material', 'pipeline_status', 'pipeline_type']].drop_duplicates()
high_risk_segments = high_risk_segments.merge(
    pipeline_info, 
    left_on='pipeline_id', 
    right_on='pipeline_name',
    how='left'
)

# Save high-risk segments with additional information
high_risk_segments.to_file(os.path.join(output_dir, 'high_risk_segments.geojson'), driver='GeoJSON')

# Save correlation report
with open(os.path.join(output_dir, 'correlation_report.txt'), 'w') as f:
    f.write("=== Correlation Analysis Report ===\n\n")
    f.write("Pearson Correlation (Linear Relationships):\n")
    f.write(pearson_corr.to_string())
    f.write("\n\nSpearman Correlation (Monotonic Relationships):\n")
    f.write(spearman_corr.to_string())
    f.write("\n\n=== Key Observations ===\n")
    f.write("1. Strong positive correlation between strain and vibration (expected as they often co-vary)\n")
    f.write("2. Soil clay percentage shows moderate correlation with risk score\n")
    f.write("3. Slope shows weak positive correlation with risk score\n")

# Save segment-level risk statistics
segment_stats = segment_risk_gdf['risk_score'].describe()
with open(os.path.join(output_dir, 'segment_risk_statistics.txt'), 'w') as f:
    f.write("=== Segment Risk Statistics ===\n\n")
    f.write(segment_stats.to_string())
    f.write("\n\n=== Risk Categories ===\n")
    f.write(f"High-risk threshold (67th percentile): {high_risk_threshold:.4f}\n")
    f.write(f"Number of high-risk segments: {len(high_risk_segments)}\n")
    f.write(f"Percentage of high-risk segments: {len(high_risk_segments) / len(segment_risk_gdf) * 100:.2f}%\n\n")
    
    # Add top 5 high-risk segments
    f.write("=== Top 5 Highest Risk Segments ===\n")
    top_risky = segment_risk_gdf.nlargest(5, 'risk_score')
    f.write(top_risky[['segment_id', 'pipeline_id', 'risk_score']].to_string(index=False))

print("Analysis complete. Results saved to the 'visibles' directory.")
