# VegLib Comprehensive User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Core Functionality](#core-functionality)
5. [Advanced Analysis Modules](#advanced-analysis-modules)
6. [Data Management](#data-management)
7. [Visualization](#visualization)
8. [Examples and Tutorials](#examples-and-tutorials)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

VegLib is a comprehensive Python package designed specifically for vegetation data analysis and environmental modeling. It provides a professional-grade suite of tools for ecologists, environmental scientists, and researchers working with biodiversity and vegetation data.

### Key Features
- **Data Management & Preprocessing**: Parse vegetation survey data, integrate remote sensing, Darwin Core standards
- **Data Quality & Validation**: Spatial/temporal validation, outlier detection, coordinate checks
- **Diversity Analysis**: 15+ diversity indices, richness estimators, beta diversity
- **Multivariate Analysis**: Complete ordination suite (PCA, CA, DCA, CCA, RDA, NMDS, PCoA)
- **Advanced Clustering**: TWINSPAN, hierarchical clustering, fuzzy C-means, DBSCAN
- **Statistical Analysis**: PERMANOVA, ANOSIM, MRPP, Mantel tests, Indicator Species Analysis
- **Temporal Analysis**: Phenology modeling, trend analysis, time series decomposition
- **Spatial Analysis**: Interpolation, landscape metrics, spatial autocorrelation
- **Machine Learning**: Predictive modeling, species distribution modeling
- **Visualization**: Specialized ecological plots, interactive dashboards

---

## 2. Installation

### Basic Installation
```bash
pip install veglib
```

### Development Version
```bash
pip install git+https://github.com/mzhatim/veglib.git
```

### Dependencies
**Required:**
- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

**Optional (for extended functionality):**
- GeoPandas (spatial analysis)
- PyProj (coordinate transformations)
- Earth Engine API (remote sensing)
- Rasterio (raster data handling)

---

## 3. Getting Started

### Basic Usage
```python
import pandas as pd
from veglib import VegLib

# Initialize VegLib
veg = VegLib()

# Load your vegetation data
data = veg.load_data('vegetation_data.csv')

# Quick diversity analysis
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])
print(diversity)

# Basic ordination
pca_results = veg.pca_analysis(transform='hellinger')
veg.plot_ordination(pca_results)
```

### Quick Analysis Functions
```python
from veglib import quick_diversity_analysis, quick_ordination, quick_clustering, quick_elbow_analysis

# Quick functions for immediate results
diversity = quick_diversity_analysis(data, species_cols=['sp1', 'sp2', 'sp3'])
ordination = quick_ordination(data, method='pca')
clusters = quick_clustering(data, n_clusters=3, method='kmeans')

# Quick elbow analysis to determine optimal clusters
elbow_results = quick_elbow_analysis(data, max_k=10, plot_results=True)
optimal_k = elbow_results['recommendations']['consensus']
print(f"Recommended number of clusters: {optimal_k}")
```

---

## 4. Core Functionality

### 4.1 VegLib Main Class

The core `VegLib` class provides the primary interface for vegetation data analysis.

```python
from veglib import VegLib

veg = VegLib()
```

### 4.2 Data Loading and Management

#### Loading Data
```python
# Load from various formats
data = veg.load_data('data.csv', format_type='csv')
data = veg.load_data('data.xlsx', format_type='excel')
data = veg.load_data('data.txt', format_type='txt')

# Specify species columns
data = veg.load_data('data.csv', species_cols=['species1', 'species2', 'species3'])
```

#### Data Preprocessing
```python
# Clean species names
veg.standardize_species_names('species_column')

# Filter rare species
veg.filter_rare_species(min_occurrences=3, min_abundance=0.5)

# Get summary statistics
stats = veg.summary_statistics()
print(stats)
```

### 4.3 Diversity Analysis

#### Calculate Multiple Indices
```python
# Calculate common diversity indices
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness', 'evenness'])

# Available indices: shannon, simpson, richness, evenness
print(diversity)
```

#### Rarefaction Analysis
```python
# Generate rarefaction curves
rarefaction = veg.rarefaction_curve(sample_sizes=[10, 50, 100, 200])

# Plot species accumulation curves
fig = veg.plot_species_accumulation(rarefaction)
```

### 4.4 Multivariate Analysis

#### Principal Component Analysis
```python
# PCA with different transformations
pca_results = veg.pca_analysis(
    transform='hellinger',  # or 'log', 'sqrt', 'standardize'
    n_components=4
)

# Access results
print("Explained variance:", pca_results['explained_variance_ratio'])
print("Site scores:", pca_results['scores'])
print("Species loadings:", pca_results['loadings'])
```

#### Non-metric Multidimensional Scaling
```python
# NMDS analysis
nmds_results = veg.nmds_analysis(
    distance_metric='bray_curtis',  # or 'euclidean', 'jaccard'
    n_dimensions=2
)

print("Stress value:", nmds_results['stress'])
```

### 4.5 Clustering Analysis

#### Hierarchical Clustering
```python
# Hierarchical clustering
hier_results = veg.hierarchical_clustering(
    distance_metric='bray_curtis',
    linkage_method='average',
    n_clusters=5
)

# Plot dendrogram
fig = veg.plot_cluster_dendrogram(hier_results)
```

#### K-means Clustering
```python
# K-means clustering
kmeans_results = veg.kmeans_clustering(
    n_clusters=4,
    transform='hellinger'
)

print("Cluster labels:", kmeans_results['cluster_labels'])
```

#### Indicator Species Analysis
```python
# Find indicator species for clusters
indicator_results = veg.indicator_species_analysis(kmeans_results['cluster_labels'])
print(indicator_results)
```

---

## 5. Advanced Analysis Modules

### 5.1 Diversity Analyzer

The `DiversityAnalyzer` class provides comprehensive diversity analysis beyond basic indices.

```python
from veglib import DiversityAnalyzer

div_analyzer = DiversityAnalyzer()

# Calculate all available indices
all_diversity = div_analyzer.calculate_all_indices(species_data)

# Available indices include:
# - shannon, simpson, simpson_inv, richness, evenness
# - fisher_alpha, berger_parker, mcintosh, brillouin
# - menhinick, margalef, chao1, ace, jack1, jack2
```

#### Hill Numbers
```python
# Calculate Hill numbers (orders of diversity)
hill_numbers = div_analyzer.hill_numbers(
    species_data, 
    q_values=[0, 1, 2, 0.5, 1.5]
)
```

#### Beta Diversity
```python
# Beta diversity analysis
beta_whittaker = div_analyzer.beta_diversity(species_data, method='whittaker')
beta_sorensen = div_analyzer.beta_diversity(species_data, method='sorensen')
beta_jaccard = div_analyzer.beta_diversity(species_data, method='jaccard')
```

### 5.2 Multivariate Analyzer

Advanced multivariate analysis with complete ordination suite.

```python
from veglib import MultivariateAnalyzer

mv_analyzer = MultivariateAnalyzer()
```

#### Correspondence Analysis
```python
# Correspondence Analysis
ca_results = mv_analyzer.correspondence_analysis(
    species_data,
    scaling=1  # or 2
)
```

#### Detrended Correspondence Analysis
```python
# DCA - useful for long ecological gradients
dca_results = mv_analyzer.detrended_correspondence_analysis(
    species_data,
    segments=26
)

print("Gradient lengths:", dca_results['gradient_lengths'])
```

#### Canonical Correspondence Analysis
```python
# CCA - constrained ordination with environmental data
cca_results = mv_analyzer.canonical_correspondence_analysis(
    species_data,
    environmental_data,
    scaling=1
)

# Environmental vectors
print("Environmental scores:", cca_results['env_scores'])
```

#### Redundancy Analysis
```python
# RDA - linear constrained ordination
rda_results = mv_analyzer.redundancy_analysis(
    species_data,
    environmental_data
)
```

#### Principal Coordinates Analysis
```python
# PCoA - metric multidimensional scaling
pcoa_results = mv_analyzer.principal_coordinates_analysis(
    species_data,
    distance_metric='bray_curtis'
)
```

#### Environmental Fitting
```python
# Fit environmental vectors to ordination
env_fit = mv_analyzer.environmental_fitting(
    ordination_results['site_scores'],
    environmental_data,
    method='vector'
)

print("R-squared values:", env_fit['r_squared'])
```

### 5.3 Vegetation Clustering

Advanced clustering methods including TWINSPAN.

```python
from veglib import VegetationClustering

clustering = VegetationClustering()
```

#### TWINSPAN Analysis
```python
# Two-Way Indicator Species Analysis
twinspan_results = clustering.twinspan(
    species_data,
    cut_levels=[0, 2, 5, 10, 20],
    max_divisions=6,
    min_group_size=5
)

print("Site classification:", twinspan_results['site_classification'])
print("Classification tree:", twinspan_results['classification_tree'])
```

#### Fuzzy C-means Clustering
```python
# Fuzzy clustering for gradual boundaries
fuzzy_results = clustering.fuzzy_cmeans_clustering(
    species_data,
    n_clusters=4,
    fuzziness=2.0
)

print("Membership matrix:", fuzzy_results['membership_matrix'])
```

#### DBSCAN for Core Communities
```python
# Density-based clustering for identifying core communities
dbscan_results = clustering.dbscan_clustering(
    species_data,
    eps=0.5,
    min_samples=5
)
```

#### Comprehensive Elbow Analysis
```python
# Comprehensive elbow analysis with multiple algorithms
elbow_results = veg.elbow_analysis(
    k_range=range(1, 15),
    methods=['knee_locator', 'derivative', 'variance_explained', 'distortion_jump'],
    transform='hellinger',
    plot_results=True
)

print("Elbow points detected:")
for method, k_value in elbow_results['elbow_points'].items():
    print(f"  {method}: k = {k_value}")

print(f"Consensus recommendation: k = {elbow_results['recommendations']['consensus']}")
print(f"Confidence: {elbow_results['recommendations']['confidence']:.2f}")

# Quick elbow analysis for rapid results
optimal_k = veg.quick_elbow_analysis(max_k=10)
print(f"Quick recommendation: k = {optimal_k}")
```

#### Available Elbow Detection Methods
- **knee_locator**: Kneedle algorithm for automatic knee/elbow detection (Satopaa et al., 2011)
- **derivative**: Second derivative maximum for curvature detection
- **variance_explained**: Point where additional clusters explain <10% more variance
- **distortion_jump**: Jump method based on distortion changes (Sugar & James, 2003)
- **l_method**: L-method for determining number of clusters (Salvador & Chan, 2004)

#### Clustering Validation
```python
# Traditional clustering validation
optimal_k = clustering.optimal_k_analysis(
    species_data,
    k_range=range(2, 11),
    methods=['elbow', 'silhouette', 'gap']
)

print("Recommended number of clusters:", optimal_k['recommendations'])
```

### 5.4 Ecological Statistics

Comprehensive statistical tests for ecological data.

```python
from veglib import EcologicalStatistics

stats_analyzer = EcologicalStatistics()
```

#### PERMANOVA
```python
# Permutational MANOVA
distance_matrix = stats_analyzer.calculate_distance_matrix(species_data, 'bray_curtis')
permanova_results = stats_analyzer.permanova(
    distance_matrix,
    groups=['group1', 'group2', 'group1', 'group2'],
    permutations=999
)

print("F-statistic:", permanova_results['F_statistic'])
print("p-value:", permanova_results['p_value'])
```

#### Mantel Tests
```python
# Mantel test for correlation between matrices
mantel_results = stats_analyzer.mantel_test(
    distance_matrix1,
    distance_matrix2,
    permutations=999
)
```

### 5.5 Temporal Analysis

Time series analysis for vegetation dynamics.

```python
from veglib import TemporalAnalyzer

temporal = TemporalAnalyzer()
```

#### Phenology Modeling
```python
# Model phenological patterns
phenology_results = temporal.phenology_modeling(
    time_series_data,
    time_col='date',
    response_col='flowering_intensity',
    model_type='sigmoid',
    species_col='species'
)
```

#### Trend Analysis
```python
# Detect trends in vegetation data
trend_results = temporal.trend_analysis(
    time_series_data,
    method='mann_kendall',
    alpha=0.05
)
```

### 5.6 Spatial Analysis

Spatial analysis and landscape ecology.

```python
from veglib import SpatialAnalyzer

spatial = SpatialAnalyzer()
```

#### Spatial Interpolation
```python
# Interpolate vegetation data across space
interpolation = spatial.spatial_interpolation(
    point_data,
    x_col='longitude',
    y_col='latitude',
    z_col='species_richness',
    method='idw',
    grid_resolution=0.01
)
```

#### Landscape Metrics
```python
# Calculate landscape-level metrics
landscape_metrics = spatial.landscape_metrics_analysis(
    raster_data,
    metrics=['patch_density', 'edge_density', 'contagion']
)
```

---

## 6. Data Management

### 6.1 Data Parsers

```python
from veglib.data_management import VegetationDataParser, TurbovegParser

# Parse various vegetation data formats
parser = VegetationDataParser()
data = parser.parse_csv('vegetation_survey.csv')

# Parse Turboveg data
turboveg_parser = TurbovegParser()
turboveg_data = turboveg_parser.parse_turboveg_export('turboveg_export.txt')
```

### 6.2 Remote Sensing Integration

```python
from veglib.data_management import RemoteSensingAPI

# Integrate with remote sensing data
rs_api = RemoteSensingAPI()

# Get NDVI data for coordinates
ndvi_data = rs_api.get_ndvi_time_series(
    coordinates=[(lon1, lat1), (lon2, lat2)],
    start_date='2020-01-01',
    end_date='2023-12-31',
    satellite='landsat8'
)
```

### 6.3 Data Standardization

```python
from veglib.data_management import DataStandardizer

standardizer = DataStandardizer()

# Standardize species names
standardized_data = standardizer.standardize_species_names(
    data,
    species_col='species',
    method='fuzzy_match'
)

# Apply data transformations
transformed_data = standardizer.apply_transformations(
    data,
    methods=['hellinger', 'wisconsin'],
    target='species'
)
```

### 6.4 Darwin Core Standards

```python
from veglib.data_management import DarwinCoreHandler

darwin_handler = DarwinCoreHandler()

# Convert to Darwin Core format
darwin_data = darwin_handler.convert_to_darwin_core(
    vegetation_data,
    mapping_config='standard_mapping.yaml'
)

# Validate Darwin Core compliance
validation_report = darwin_handler.validate_darwin_core(darwin_data)
```

---

## 7. Visualization

### 7.1 Basic Plotting

```python
# Plot diversity indices
diversity_plot = veg.plot_diversity(diversity_results, 'shannon')

# Plot ordination results
ordination_plot = veg.plot_ordination(
    pca_results,
    color_by=group_variable
)

# Plot clustering dendrogram
dendrogram = veg.plot_cluster_dendrogram(hier_results)
```

### 7.2 Interactive Visualization

```python
from veglib import InteractiveVisualizer

viz = InteractiveVisualizer()

# Create interactive dashboard
dashboard = viz.create_dashboard(
    data=species_data,
    environmental_data=env_data,
    include_plots=['ordination', 'diversity', 'clustering']
)

# Export interactive plots
viz.export_interactive_plot(ordination_results, 'ordination.html')
```

### 7.3 Report Generation

```python
from veglib import ReportGenerator

report_gen = ReportGenerator()

# Generate comprehensive analysis report
report = report_gen.generate_full_report(
    species_data=species_data,
    environmental_data=env_data,
    analyses=['diversity', 'ordination', 'clustering'],
    output_format='html'
)
```

---

## 8. Examples and Tutorials

### 8.1 Complete Vegetation Analysis Workflow

```python
import pandas as pd
from veglib import VegLib

# 1. Initialize and load data
veg = VegLib()
data = veg.load_data('vegetation_survey.csv')

# 2. Data preprocessing
veg.standardize_species_names('species')
veg.filter_rare_species(min_occurrences=3)

# 3. Diversity analysis
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])
rarefaction = veg.rarefaction_curve()

# 4. Multivariate analysis
pca_results = veg.pca_analysis(transform='hellinger')
nmds_results = veg.nmds_analysis(distance_metric='bray_curtis')

# 5. Clustering
kmeans_results = veg.kmeans_clustering(n_clusters=4)
indicators = veg.indicator_species_analysis(kmeans_results['cluster_labels'])

# 6. Statistical tests
from veglib import EcologicalStatistics
stats = EcologicalStatistics()
distance_matrix = stats.calculate_distance_matrix(veg.species_matrix)
permanova = stats.permanova(distance_matrix, groups)

# 7. Visualization
veg.plot_diversity(diversity, 'shannon')
veg.plot_ordination(pca_results, color_by=kmeans_results['cluster_labels'])
veg.plot_cluster_dendrogram(hier_results)

# 8. Export results
veg.export_results(diversity, 'diversity_results.csv')
veg.export_results(pca_results, 'pca_results.csv')
```

### 8.2 Advanced TWINSPAN Analysis

```python
from veglib import VegetationClustering

# Initialize clustering
clustering = VegetationClustering()

# TWINSPAN classification
twinspan_results = clustering.twinspan(
    species_data,
    cut_levels=[0, 1, 2, 5, 10, 20],
    max_divisions=8,
    min_group_size=3
)

# Examine results
print("Number of final groups:", len(twinspan_results['classification_tree']['groups']))
print("Site classification:", twinspan_results['site_classification'])

# Examine indicator species for each division
for division in twinspan_results['classification_tree']['divisions']:
    print(f"\nDivision {division['division_id']}:")
    print(f"Eigenvalue: {division['eigenvalue']:.3f}")
    for indicator in division['indicator_species'][:3]:  # Top 3
        print(f"  - {indicator['species']}: diff = {indicator['frequency_difference']:.3f}")
```

### 8.3 Comprehensive Diversity Analysis

```python
from veglib import DiversityAnalyzer

div_analyzer = DiversityAnalyzer()

# Calculate all diversity indices
all_diversity = div_analyzer.calculate_all_indices(species_data)

# Hill numbers for different orders
hill_numbers = div_analyzer.hill_numbers(species_data, q_values=[0, 0.5, 1, 1.5, 2])

# Beta diversity analysis
beta_sorensen = div_analyzer.beta_diversity(species_data, method='sorensen')
beta_jaccard = div_analyzer.beta_diversity(species_data, method='jaccard')

# Compare sites using beta diversity
print("Average Sørensen dissimilarity:", beta_sorensen.mean().mean())
print("Average Jaccard dissimilarity:", beta_jaccard.mean().mean())
```

### 8.4 Environmental Gradient Analysis

```python
from veglib import MultivariateAnalyzer

mv_analyzer = MultivariateAnalyzer()

# Canonical Correspondence Analysis
cca_results = mv_analyzer.cca_analysis(species_data, environmental_data)

# Environmental fitting to unconstrained ordination
dca_results = mv_analyzer.detrended_correspondence_analysis(species_data)
env_fit = mv_analyzer.environmental_fitting(
    dca_results['site_scores'], 
    environmental_data
)

# Identify significant environmental variables
significant_vars = {var: r2 for var, r2 in env_fit['r_squared'].items() if r2 > 0.1}
print("Significant environmental variables:", significant_vars)
```

---

## 9. API Reference

### 9.1 Core Classes

#### VegLib
Main class providing comprehensive vegetation analysis functionality.

**Methods:**
- `load_data(filepath, format_type, **kwargs)` - Load data from various formats
- `calculate_diversity(indices)` - Calculate diversity indices
- `pca_analysis(transform, n_components)` - Principal Component Analysis
- `nmds_analysis(distance_metric, n_dimensions)` - NMDS ordination
- `hierarchical_clustering(distance_metric, linkage_method, n_clusters)` - Hierarchical clustering
- `kmeans_clustering(n_clusters, transform)` - K-means clustering
- `indicator_species_analysis(clusters)` - Find indicator species

#### DiversityAnalyzer
Specialized class for diversity analysis.

**Methods:**
- `calculate_all_indices(data)` - Calculate all available diversity indices
- `hill_numbers(data, q_values)` - Calculate Hill numbers
- `beta_diversity(data, method)` - Beta diversity analysis
- `shannon_diversity(data)` - Shannon diversity index
- `simpson_diversity(data)` - Simpson diversity index
- `chao1_estimator(data)` - Chao1 richness estimator

#### MultivariateAnalyzer
Advanced multivariate analysis methods.

**Methods:**
- `pca_analysis(data, transform, n_components)` - Enhanced PCA
- `nmds_analysis(data, distance_metric, n_dimensions)` - Enhanced NMDS
- `correspondence_analysis(data, scaling)` - Correspondence Analysis
- `detrended_correspondence_analysis(data, segments)` - DCA
- `canonical_correspondence_analysis(species_data, env_data, scaling)` - CCA
- `redundancy_analysis(species_data, env_data)` - RDA
- `environmental_fitting(ordination_scores, env_data, method)` - Fit environmental vectors

#### VegetationClustering
Advanced clustering methods for vegetation classification.

**Methods:**
- `twinspan(data, cut_levels, max_divisions, min_group_size)` - TWINSPAN analysis
- `fuzzy_cmeans_clustering(data, n_clusters, fuzziness)` - Fuzzy C-means
- `dbscan_clustering(data, eps, min_samples)` - DBSCAN clustering
- `gaussian_mixture_clustering(data, n_components)` - Gaussian Mixture Models
- `optimal_k_analysis(data, k_range, methods)` - Find optimal number of clusters

#### EcologicalStatistics
Statistical tests for ecological data.

**Methods:**
- `permanova(distance_matrix, groups, permutations)` - PERMANOVA test
- `anosim(distance_matrix, groups, permutations)` - ANOSIM test  
- `mantel_test(matrix1, matrix2, permutations)` - Mantel test
- `mrpp(distance_matrix, groups, permutations)` - MRPP test

### 9.2 Quick Functions

- `quick_diversity_analysis(data, species_cols)` - Rapid diversity analysis
- `quick_ordination(data, species_cols, method)` - Quick ordination
- `quick_clustering(data, species_cols, n_clusters, method)` - Quick clustering

### 9.3 Data Management Classes

#### VegetationDataParser
Parse vegetation survey data from various formats.

#### TurbovegParser
Specialized parser for Turboveg data format.

#### RemoteSensingAPI
Integration with remote sensing data sources.

#### DataStandardizer
Standardize and transform ecological data.

#### DarwinCoreHandler
Handle Darwin Core biodiversity data standards.

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Import Errors
```python
# If you get import errors for optional dependencies
import warnings
warnings.filterwarnings('ignore', category=ImportWarning)

# Check what's available
from veglib import show_versions
show_versions()
```

#### Memory Issues with Large Datasets
```python
# For large datasets, use data chunking or filtering
veg.filter_rare_species(min_occurrences=5)  # Reduce species
data_subset = data.sample(n=1000)  # Random subset
```

#### Convergence Issues in Ordination
```python
# Increase max iterations for NMDS
nmds_results = veg.nmds_analysis(
    distance_metric='bray_curtis',
    n_dimensions=2,
    max_iterations=500  # Increase iterations
)

# Check stress value - should be < 0.2
print("Stress:", nmds_results['stress'])
```

### 10.2 Data Format Requirements

#### Species Data Format
Data should be in site-by-species matrix format:

```
Site_ID | Species1 | Species2 | Species3 | ...
--------|----------|----------|----------|----
Site1   |    5     |    3     |    0     | ...
Site2   |    2     |    7     |    1     | ...
Site3   |    0     |    4     |    8     | ...
```

#### Environmental Data Format
Environmental data should have matching site IDs:

```
Site_ID | Temperature | Precipitation | pH  | ...
--------|-------------|---------------|-----|----
Site1   |    25.3     |     850       | 6.2 | ...
Site2   |    23.8     |     920       | 5.9 | ...
Site3   |    27.1     |     780       | 6.8 | ...
```

### 10.3 Performance Optimization

```python
# For large datasets, consider:
# 1. Use appropriate data transformations
data_transformed = veg._transform_data(data, 'hellinger')

# 2. Reduce dimensionality first
pca_results = veg.pca_analysis(n_components=10)

# 3. Use appropriate distance metrics
# Bray-Curtis for abundance data, Jaccard for presence/absence
```

### 10.4 Getting Help

- **Documentation**: Check function docstrings with `help(function_name)`
- **Examples**: Use `veg.summary_statistics()` to understand your data
- **Validation**: Always check ordination stress values and clustering validation metrics
- **Citations**: Use `citation()` function for proper citation information

---

## Citation

To cite VegLib in publications:

```
Hatim, M.Z. (2025). VegLib: A comprehensive Python package for vegetation 
data analysis and environmental modeling. Version 1.0.0.
```

BibTeX:
```bibtex
@software{veglib2025,
    author = {Hatim, Mohamed Z.},
    title = {VegLib: A comprehensive Python package for vegetation data analysis and environmental modeling},
    year = {2025},
    version = {1.0.0},
    url = {https://github.com/mzhatim/veglib}
}
```

---

**Copyright (c) 2025 Mohamed Z. Hatim**  
**License: MIT**