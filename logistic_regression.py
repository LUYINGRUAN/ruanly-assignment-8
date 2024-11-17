import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

# Directory to save results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Function to generate ellipsoid clusters with a shift
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                   [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and apply the shift
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2 += np.array([distance, distance])  # Shift cluster along both axes
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

# Function to perform experiments
def do_experiments(start, end, step_num):
    # Set up experiment parameters
    shift_distances = np.linspace(start=start, stop=end, num=step_num)  # Range of shift distances
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list, loss_list, margin_widths = [], [], [], []
    sample_data = {}  # Store sample datasets and models for visualization

    n_samples = 8
    n_cols = 2  # Fixed number of columns
    n_rows = (step_num + n_cols - 1) // n_cols  # Calculate rows needed
    plt.figure(figsize=(20, n_rows * 10))  # Adjust figure height based on rows

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        
        # Fit logistic regression and extract parameters
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        # Calculate logistic loss
        z = beta0 + beta1 * X[:, 0] + beta2 * X[:, 1]
        y_pred_prob = 1 / (1 + np.exp(-z))
        logistic_loss = -np.mean(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
        loss_list.append(logistic_loss)

        # Calculate slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Plot the dataset
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', alpha=0.7, label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', alpha=0.7, label='Class 1')
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_values = np.linspace(x_min, x_max, 200)
        y_values = slope * x_values + intercept
        plt.plot(x_values, y_values, color='black', label='Decision Boundary')

        # Calculate margin width between 70% confidence contours
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot 1: Parameters vs. Shift Distance
    plt.figure(figsize=(18, 15))

    # Beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o', color='blue')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o', color='green')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o', color='red')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Slope
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o', color='purple')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Logistic Loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o', color='magenta')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Margin Width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', color='cyan')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
