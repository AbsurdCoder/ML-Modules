# data_analysis_and_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


class DataPreprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def display_head(self):
        print("Displaying the first few rows of the dataset:")
        print(self.data.head())

    def missing_values_summary(self):
        print("Summary of missing values:")
        print(self.data.isnull().sum())

    def fill_missing_values(self, strategy='mean'):
        print(f"Filling missing values using {strategy} strategy.")
        imputer = SimpleImputer(strategy=strategy)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def visualize_data(self):
        print("Visualizing data distributions and correlations.")
        self.data.hist(bins=30, figsize=(15, 10))
        plt.show()
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.show()

    def normality_test(self, column, method='dagostino'):
        print(f"Performing normality test on {column} using {method}.")
        if method == 'dagostino':
            k2, p = stats.normaltest(self.data[column])
            print(f"D'Agostino and Pearson’s test p-value: {p}")
            if p < 0.05:
                print(f"The data in {column} is not normally distributed.")
            else:
                print(f"The data in {column} is normally distributed.")
        elif method == 'shapiro':
            stat, p = stats.shapiro(self.data[column])
            print(f"Shapiro-Wilk test p-value: {p}")
            if p < 0.05:
                print(f"The data in {column} is not normally distributed.")
            else:
                print(f"The data in {column} is normally distributed.")
        else:
            raise ValueError("Method must be 'dagostino' or 'shapiro'.")

    def correlation_test(self, col1, col2, method='pearson'):
        print(f"Performing {method} correlation test between {col1} and {col2}.")
        if method == 'pearson':
            corr, p = stats.pearsonr(self.data[col1], self.data[col2])
        elif method == 'spearman':
            corr, p = stats.spearmanr(self.data[col1], self.data[col2])
        elif method == 'kendall':
            corr, p = stats.kendalltau(self.data[col1], self.data[col2])
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'.")
        print(f"Correlation: {corr}, p-value: {p}")

    def t_test(self, col, group_col, group1, group2):
        print(f"Performing t-test between {group1} and {group2} for {col}.")
        group1_data = self.data[self.data[group_col] == group1][col]
        group2_data = self.data[self.data[group_col] == group2][col]
        t_stat, p = stats.ttest_ind(group1_data, group2_data)
        print(f"t-statistic: {t_stat}, p-value: {p}")

    def chi_square_test(self, col1, col2):
        print(f"Performing chi-square test between {col1} and {col2}.")
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-square: {chi2}, p-value: {p}, Degrees of freedom: {dof}")

    def get_preprocessed_data(self):
        return self.data


class DataNormalization:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def standard_scaler(self):
        print("Applying standard scaling.")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        return pd.DataFrame(scaled_data, columns=self.data.columns)

    def min_max_scaler(self):
        print("Applying min-max scaling.")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data)
        return pd.DataFrame(scaled_data, columns=self.data.columns)


class DistributionCheck:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check_distribution(self, graphical=False):
        """
        Check the distribution of each feature in the dataset.
        If graphical is True, display histograms, box plots, and QQ plots.
        """
        for column in self.data.columns:
            print(f"Checking distribution for {column}:")
            
            # Summary statistics
            summary = self.data[column].describe()
            print(summary)
            
            if graphical:
                self._plot_distribution(self.data[column], column)

            # Normality test results
            shapiro_test = stats.shapiro(self.data[column].dropna())
            print(f"Shapiro-Wilk test p-value for {column}: {shapiro_test.pvalue}")
            dagostino_test = stats.normaltest(self.data[column].dropna())
            print(f"D'Agostino and Pearson’s test p-value for {column}: {dagostino_test.pvalue}")
            
            print("\n")
            
    def _plot_distribution(self, data, column_name):
        """
        Plot the distribution of a feature including histogram, box plot, and QQ plot.
        """
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(data, kde=True)
        plt.title(f'Histogram of {column_name}')
        
        # Box plot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=data)
        plt.title(f'Box plot of {column_name}')
        
        # QQ plot
        plt.subplot(1, 3, 3)
        stats.probplot(data.dropna(), dist="norm", plot=plt)
        plt.title(f'QQ plot of {column_name}')
        
        plt.tight_layout()
        plt.show()


class KMeansClustering:
    def __init__(self, data: pd.DataFrame, n_clusters: int):
        self.data = data
        self.n_clusters = n_clusters

    def fit_predict(self):
        print(f"Applying KMeans with {self.n_clusters} clusters.")
        model = KMeans(n_clusters=self.n_clusters)
        self.labels = model.fit_predict(self.data)
        return self.labels

    def plot_clusters(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=self.labels, cmap='viridis')
        plt.title('KMeans Clustering')
        plt.show()


class AgglomerativeClusteringModel:
    def __init__(self, data: pd.DataFrame, n_clusters: int):
        self.data = data
        self.n_clusters = n_clusters

    def fit_predict(self):
        print(f"Applying Agglomerative Clustering with {self.n_clusters} clusters.")
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.labels = model.fit_predict(self.data)
        return self.labels

    def plot_clusters(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=self.labels, cmap='viridis')
        plt.title('Agglomerative Clustering')
        plt.show()


class DBSCANClustering:
    def __init__(self, data: pd.DataFrame, eps: float, min_samples: int):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self):
        print(f"Applying DBSCAN with eps={self.eps} and min_samples={self.min_samples}.")
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = model.fit_predict(self.data)
        return self.labels

    def plot_clusters(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=self.labels, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.show()


class AccuracyMetrics:
    def __init__(self, y_true, y_pred, y_prob=None):
        """
        Initialize the class with true labels and predicted labels.
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities (optional, for ROC/AUC)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        
    def accuracy(self):
        """
        Calculate and print the accuracy score.
        """
        score = accuracy_score(self.y_true, self.y_pred)
        print(f"Accuracy: {score:.4f}")
        return score

    def precision(self, average='binary'):
        """
        Calculate and print the precision score.
        average: 'binary', 'micro', 'macro', 'weighted' (default: 'binary')
        """
        score = precision_score(self.y_true, self.y_pred, average=average)
        print(f"Precision: {score:.4f}")
        return score

    def recall(self, average='binary'):
        """
        Calculate and print the recall score.
        average: 'binary', 'micro', 'macro', 'weighted' (default: 'binary')
        """
        score = recall_score(self.y_true, self.y_pred, average=average)
        print(f"Recall: {score:.4f}")
        return score

    def f1(self, average='binary'):
        """
        Calculate and print the F1 score.
        average: 'binary', 'micro', 'macro', 'weighted' (default: 'binary')
        """
        score = f1_score(self.y_true, self.y_pred, average=average)
        print(f"F1 Score: {score:.4f}")
        return score

    def confusion_matrix(self, labels=None):
        """
        Display the confusion matrix.
        labels: list of labels (default: None)
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        print(f"Confusion Matrix:\n{cm}")
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def roc_auc(self, pos_label=1):
        """
        Calculate and plot the ROC curve and AUC score.
        pos_label: label considered as the positive class (default: 1)
        """
        if self.y_prob is None:
            raise ValueError("y_prob is required for ROC/AUC calculations.")
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        print(f"AUC Score: {roc_auc:.4f}")
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def print_all_metrics(self, average='binary', labels=None, pos_label=1):
        """
        Print all accuracy metrics.
        """
        print("Calculating accuracy metrics...")
        self.accuracy()
        self.precision(average)
        self.recall(average)
        self.f1(average)
        self.confusion_matrix(labels)
        
        if self.y_prob is not None:
            self.roc_auc(pos_label)
