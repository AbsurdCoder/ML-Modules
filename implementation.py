from cluster_class import (
    DataPreprocessing,
    DataNormalization,
    DistributionCheck,
    KMeansClustering,
    AgglomerativeClusteringModel,
    DBSCANClustering,
    AccuracyMetrics
)

import pandas as pd
# Load your data
data = pd.read_csv('your_dataset.csv')

# Data Preprocessing
preprocessor = DataPreprocessing(data)
preprocessor.display_head()
preprocessor.missing_values_summary()
preprocessor.fill_missing_values()
preprocessor.visualize_data()
preprocessor.normality_test('your_column_name', method='shapiro')
preprocessor.correlation_test('column1', 'column2')
preprocessor.t_test('numerical_column', 'categorical_column', 'group1', 'group2')
preprocessor.chi_square_test('categorical_column1', 'categorical_column2')
processed_data = preprocessor.get_preprocessed_data()

# Data Normalization
normalizer = DataNormalization(processed_data)
normalized_data = normalizer.standard_scaler()  # or normalizer.min_max_scaler()

# Distribution Check
distribution_checker = DistributionCheck(normalized_data)
distribution_checker.check_distribution(graphical=True)

# Clustering
kmeans = KMeansClustering(normalized_data, n_clusters=3)
labels_kmeans = kmeans.fit_predict()
kmeans.plot_clusters()

agglomerative = AgglomerativeClusteringModel(normalized_data, n_clusters=3)
labels_agglo = agglomerative.fit_predict()
agglomerative.plot_clusters()

dbscan = DBSCANClustering(normalized_data, eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict()
dbscan.plot_clusters()

# Assuming you have a classifier and predictions
# y_true = [...]  # True labels
# y_pred = [...]  # Predicted labels
# y_prob = [...]  # Predicted probabilities

# Accuracy Metrics
# accuracy_metrics = AccuracyMetrics(y_true, y_pred, y_prob)
# accuracy_metrics.print_all_metrics(labels=[0, 1], pos_label=1)
