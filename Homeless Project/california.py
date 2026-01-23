from multiprocessing import Value
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
import shap
import os


class california:
    """
    This is a data analysis object created to analyse a Kaggle dataset by the name of "California Homelessness Prediction Challenge".
    The creator of this dataset notes that HUD Point-in-Time counts of homelessness are not raw rates that could be used in conjunction
    with other demographic rates of a population to examine the relationship between each. Therefore, he constructed a dataset
    spanning 10 californian counties, and 200 specific regions containing 48 of the 58 HUD Point-in-Time demographic category raets for each area obtained from census-based demographic
    profiles mathcing HUD geographic boundaries in California, and the corresponding
    rates of homelessness calculated from HUD Point-in-Time homelessness counts. This allows for the explorartion of whether demographic rates
    in different areas could be used to estimate rates of homelessness. Moreover, using explainable machine learning models
    such as the linear regression models which are trained in this data analysis object could provide insight into
    which sets of demographic features contain the most predictive value when attempting to estimate rates of homelessness
    in different regions.

    This Data Analysis object provides insight into the kaggle dataset in the following ways:
     - The question of which demographic features contain most predictive value is examined accross three groupings of demographics:
       - race
       - age
       - all the features in the dataset
    - You may view the Top two Principal components found through PCA to inspect which features of the dataset result in the most variability in homelessness
    - You may inspect the Linear models' coefficients' values to check out which features had the greatest weights for predicting
    - You may display the pearson correlation coefficients of the features of any of the above demographic groupings to the rate of homelessness
    - You may display the homelessness statistics of the training dataset obtained from kaggle
    - You may display stats for each of the demographic groupings covered above.
    - You may display the shapley values for each of the features used by each of the demographic groupings' models in prediction:
    this provides insight into which features contributed the most to a model's prediciton and is based on a game theoretic appraoch
    for understanding the contribution of a feature to the predictive ability of a model.

    Link to dataset: https://www.kaggle.com/competitions/california-homelessness-prediction-challenge/overview
    """

    def __init__(self, data_path: str):

        assert type(data_path) == str
        assert os.path.isfile(os.path.join(data_path, "train.csv"))

        self.data_path = data_path
        self.data_train = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        # self.data_test = pd.read_csv(os.path.join(self.data_path, "test.csv"))
        self.raw_data = self.data_train
        self.X_raw = self.prepare_data(self.raw_data.iloc[:, 2:].to_numpy())
        self.y_raw = self.raw_data.iloc[:, 1].to_numpy()

        # assert len(self.data_train.columns) == len(self.data_test.columns)
        # assert len(self.data_train.columns) == len(self.raw_data.columns)
        any_nan_in_df = self.raw_data.isna().any().any()
        assert not any_nan_in_df
        # print(f"Any NaN in the DataFrame: {any_nan_in_df}")

        self.age_df = self.extract_df(self.raw_data, "age")
        self.race_df = self.extract_df(self.raw_data, "race")
        self.age_X = self.prepare_data(self.age_df.iloc[:, 2:].to_numpy())
        self.race_X = self.prepare_data(self.race_df.iloc[:, 2:].to_numpy())

        self.cali_model = self.train_linear_model(self.X_raw, self.y_raw)
        self.age_model = self.train_linear_model(self.age_X, self.y_raw)
        self.race_model = self.train_linear_model(self.race_X, self.y_raw)

        self.cali_shap_vals, self.cali_explainer = self.explain_shap("all")
        self.age_shap_vals, self.age_explainer = self.explain_shap("age")
        self.race_shap_vals, self.race_explainer = self.explain_shap("race")

    def __repr__(self):
        return "California Homelessness Analysis Object:\n\n Contains 33 demographic features of Californian counties to be used in the exploration of the relationship\n between demographic statistics and rates of homelessness in California."

    def feature_correlations(self, demog: str):
        """
        Computes correlation coefficient of each feature in X with target y_np.

        Parameters:
            X (np.ndarray): shape (n_samples, n_features)
            y_np (np.ndarray): shape (n_samples,)

        Returns:
            sorted_corrs (np.ndarray): correlation coefficients sorted (highest to lowest absolute magnitude)
            sorted_indices (np.ndarray): feature indices sorted by correlation magnitude
        """
        if demog == "age":
            X_val, y_np_temp = self.age_X, self.y_raw
        elif demog == "race":
            X_val, y_np_temp = self.race_X, self.y_raw
        elif demog == "all":
            X_val, y_np_temp = self.X_raw, self.y_raw
        else:
            raise ValueError("Unknown demographic group selected.")
        # Compute correlations for each feature column with y

        corrs = np.array([np.corrcoef(X_val[:, i], y_np_temp)[0, 1] for i in range(X_val.shape[1])])

        # Sort by absolute correlation value, descending
        sorted_indices = np.argsort(np.abs(corrs))[::-1]
        sorted_corrs = corrs[sorted_indices]

        return sorted_corrs, sorted_indices

    def display_correlations(self, demog: str):
        """

        :param demog: Choose the type of demographics you wish to display using either "all", "race", "age".
        :return: A plot containing the pearson correlation coefficients of each of the demographic features to the rate of homelessness.
        """
        if demog == "age":
            df = self.age_df
        elif demog == "race":
            df = self.race_df
        elif demog == "all":
            df = self.raw_data
        else:
            raise ValueError("Unknown demographic group selected.")

        corr_values, corr_indices = self.feature_correlations(demog)
        corr_indices += 2
        corr_names = df.columns[corr_indices].tolist()
        colors = ["green" if val >= 0 else "red" for val in corr_values]

        plt.figure(figsize=(10, 5))

        plt.bar(corr_names, corr_values, color=colors)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Pearson Correlation Coefficient")
        plt.title("Feature Correlation With Target")

        plt.axhline(0, color='black', linewidth=1)  # horizontal line at y=0

        plt.tight_layout()
        plt.show()

    def display_homelessness_stats(self):
        """

        :return: Displays Min, Mean, Median, and Max statistics of rates of homelessness from the Kaggle Dataset.
        """
        data_max_rate = self.y_raw.max()
        data_min_rate = self.y_raw.min()
        data_avg_rate = self.y_raw.mean()
        data_median_rate = np.median(self.y_raw)

        print(f"\nMax Rate: {data_max_rate}\n")
        print(f"Min Rate: {data_min_rate}\n")
        print(f"mean Rate: {data_avg_rate}\n")
        print(f"Median Rate: {data_median_rate}\n")

    def pca_plot(self):
        """
        Principal Component Analysis (PCA) is a dimensionality-reduction method that transforms a dataset into a new set of orthogonal axes-called principal components-that capture directions of maximum variance in the data.
        Each component is a weighted combination of the original features, and the first few components usually summarize most of the dataset's overall structure.
        By projecting data onto these components, PCA reveals which features vary the most together, which are redundant, and which drive the strongest patterns across samples.
        When applied to demographic features with a target variable like homelessness rates, PCA can show which combinations of demographic factors-such as age distribution, veteran status, household size, or race-associated disparities-account for the largest differences across regions.
        Features with large weights in a principal component that aligns strongly with homelessness rates indicate demographic dimensions that co-vary with homelessness and contribute heavily to its variability; in this way,
        PCA helps uncover broad demographic patterns that underlie increases or decreases in homelessness across the dataset.

        :return: Computes the top 2 Principal components of the dataset, and plots a color mapped scatter plot of
        all the points in the dataset after mapping them to the two principal components to display how homelessness rates
        vary across the two principal components.
        """
        pca = PCA(n_components=2)  # change this to desired number of PCs
        X_pca = pca.fit_transform(self.X_raw)
        # feature names — from your dataframe
        feature_names = self.raw_data.columns

        # Number of top contributing features to display
        top_k = 5

        for i in range(pca.n_components_):
            component = pca.components_[i]

            # Get indices of top absolute loadings
            top_features_idx = np.argsort(np.abs(component))[-top_k:][::-1] + 2
            # print(np.argsort(np.abs(component))[-top_k:][::-1])
            top_features = feature_names[top_features_idx]
            print(f"\nTop features contributing to PCA{i + 1}:\n")
            print(top_features.tolist(), "\n\n")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        # 2D scatter plot (PC1 vs PC2)
        p = ax.scatter(
            X_pca[:, 0],  # PC1
            X_pca[:, 1],  # PC2
            c=np.log10(self.y_raw + 0.000001),  # color by normalized target
            cmap='viridis',
            s=60
        )

        # Axis labeling
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Colorbar to show target value scale
        fig.colorbar(p, ax=ax, label="log base 10 of Homelessness Rate")
        plt.title("Variability of Homelessness accross the PC's")
        plt.show()

    @staticmethod
    def top_k_abs_values(arr, k):
        """

        :param arr: list of a model's coefficient values.
        :param k: choice of the number of highest absolute values of model coefficients to return and the indicies of their features.
        :return: Higheset absolute values of model coefficients and the indicies of their features.
        """
        arr = np.asarray(arr)  # ensure numpy array

        # argsort sorts ascending, so we take the last k and reverse for descending
        idx = np.argsort(np.abs(arr))[-k:][::-1]

        values = arr[idx]  # NumPy advanced indexing returns the values at those indices

        return idx, values

    def display_highest_coefficient_vals(self, demog: str, k: int):
        """

        :param demog: Choose the type of demographics you wish to display using either "all", "race", "age".
        :param k: choice of the number of highest absolute values of model coefficients to display and the names of their features.
        :return: Higheset absolute values of model coefficients and the names of their features.
        """
        if demog == "age":
            df, model = self.age_df, self.age_model
        elif demog == "race":
            df, model = self.race_df, self.race_model
        elif demog == "all":
            df, model = self.raw_data, self.cali_model
        else:
            raise ValueError("Unknown demographic group selected.")

        ind, vals = self.top_k_abs_values(model.coef_, k)
        ind = [i + 2 for i in ind]
        print(df.columns[ind].tolist())
        print(np.abs(vals))

    @staticmethod
    def prepare_data(X: np.ndarray):
        """

        :param X: Dataset to be standardised
        :return: Dataset Standardised by normalizing accross each feature.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    @staticmethod
    def train_linear_model(X: np.ndarray, y: np.ndarray):
        """

        :param X: Training samples to be used in training.
        :param y: Rates of homelessness corresponding to each sample.
        :return: Trained model.
        """
        # create model
        model = LinearRegression()

        # fit (train) model
        model.fit(X, y)

        return model

    def demog_stats(self, demog: str):
        """

        :param demog: Choose the type of demographics you wish to display using either "all", "race", "age".
        :return: Dataframe with Min, Mean, Median, Max, and Standard Deviation values for each features in the chosen demographic type.
        """
        if demog == "age":
            df = self.age_df
        elif demog == "race":
            df = self.race_df
        elif demog == "all":
            df = self.X_raw
        else:
            raise ValueError("Unknown demographic group selected.")

        demog_stats = pd.DataFrame({
            'mean rate': df.iloc[:, 2:].mean(),
            'median rate': df.iloc[:, 2:].median(),
            'min rate': df.iloc[:, 2:].min(),
            'max rate': df.iloc[:, 2:].max(),
            'std rate': df.iloc[:, 2:].std()
        })
        print(demog_stats)
        return demog_stats

    def predict_linear_model(self, demog: str):
        """

        :param demog: Which demographic grouping. Choose between "all", "race", and "age".
        :return: Predictions of the model.
        """
        if demog == "age":
            X, model = self.age_x, self.age_model
        elif demog == "race":
            X, model = self.race_x, self.race_model
        elif demog == "all":
            X, model = self.X_raw, self.cali_model
        else:
            raise ValueError("Unknown demographic group selected.")

        y_pred = model.predict(X)
        return y_pred

    def Evaluate_accuracy(self, demog: str):  # To be used for notebook
        """

        :param demog: Which demographic grouping. Choose between "all", "race", and "age".
        :return: Mean Squared Error of the model in its predicitons.
        """
        if demog == "age" or demog == "race" or demog == "all":
            y_pred = self.predict_linear_model(demog)
        else:
            raise ValueError("Unknown demographic group selected.")

        error = mean_squared_error(self.y_raw, y_pred)
        print("Mean Squared Error of Model: ", error)
        return error

    @staticmethod
    def extract_df(df: pd.DataFrame, demographic: str):
        """
        Used in the __init__() function of the class to extract demographics based on two groupings: "race", and "age", and
        create dataframes for each.

        :param df: Raw Dataframe with all features.
        :param demographic: Which grouping of demographic.
        :return: Dataframe with the chosen demographic grouping's features.
        """
        if demographic == "age":
            # 1. First column name
            first_col = df.columns[0:2].to_list()

            # 2. Columns starting with "AGE" — but SORTED numerically

            def extract_age_key(col):
                """

                :param col: column name
                :return: the integer value representing the lower bound of the age interval for the chosen column col.
                """
                # col looks like: "AGE_25_30_PCT"
                # extract the two numbers using a regex
                nums = re.findall(r"\d+", col)
                return tuple(int(n) for n in nums)  # e.g. (25, 30)

            # sort AGE columns by the first number, then the second
            age_cols = sorted(
                [col for col in df.columns if col.startswith("AGE") and col != "AGE_25_PLUS_PCT"],
                key=extract_age_key
            )

            # 3. Combine them
            df_age = df[first_col + age_cols].copy()

            return df_age

        elif demographic == "race":
            # 1. First column name
            first_col = df.columns[0:2]

            first_col = first_col.to_list()
            # 2. Columns starting with "Race"
            race_cols = [col for col in df.columns if col.startswith("RACE")]

            # 3. Combine them
            df_race = df[first_col + race_cols].copy()

            return df_race

        else:
            raise ValueError("Unknown demographic group selected.")

    def explain_shap(self, demog: str):
        """
        Shapley values provide a principled way to explain a model’s predictions by attributing to each feature its fair share of contribution,
        based on how much that feature improves the model’s performance across all possible combinations of features.
        Borrowed from cooperative game theory, they treat each feature as a “player” whose value is the change in prediction when that feature joins a subset of other features.
        As a result, Shapley values capture not only each feature’s direct effect on a prediction but also its interaction effects with other features,
        offering a globally consistent measure of importance. Applied across a dataset, they reveal which features the model relies on most,
        how sensitive the model is to specific inputs, and whether the learned patterns align with domain knowledge or reflect spurious correlations.
        Overall, Shapley values serve as a powerful lens into both how a model behaves and what structure the data contains that the model is exploiting.

        :param demog: Which demographic grouping. Choose between "all", "race", and "age".
        :return: Shapley values, and explainer object to be used for plotting the Shapley values of each of the featuers for the
        different demographic groupings covered by this data analysis object
        """
        if demog == "age":
            X, model, feature_names = self.age_X, self.age_model, self.age_df.columns[2:]
        elif demog == "race":
            X, model, feature_names = self.race_X, self.race_model, self.race_df.columns[2:]
        elif demog == "all":
            X, model, feature_names = self.X_raw, self.cali_model, self.raw_data.columns[2:]
        else:
            raise ValueError("Unknown demographic group selected.")

        explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
        shap_values = explainer(X)
        return shap_values, explainer

    def shap_bar_plot(self, demog: str):
        """

        :param demog: Which demographic grouping. Choose between "all", "race", and "age".
        :return: A bar plot of the shapley values of each feature.
        """
        if demog == "age":
            shap_vals, explainer = self.age_shap_vals, self.age_explainer
        elif demog == "race":
            shap_vals, explainer = self.race_shap_vals, self.race_explainer
        elif demog == "all":
            shap_vals, explainer = self.cali_shap_vals, self.cali_explainer
        else:
            raise ValueError("Unknown demographic group selected.")

        shap.plots.bar(shap_vals)

    def shap_bar_plot_cluser(self, demog: str):
        """

        :param demog: Which demographic grouping. Choose between "all", "race", and "age".
        :return: A bar plot of the shapley values of each feature using clustering.
        """
        if demog == "age":
            shap_vals, explainer, X = self.age_shap_vals, self.age_explainer, self.age_X
        elif demog == "race":
            shap_vals, explainer, X = self.race_shap_vals, self.race_explainer, self.race_X
        elif demog == "all":
            shap_vals, explainer, X = self.cali_shap_vals, self.cali_explainer, self.X_raw
        else:
            raise ValueError("Unknown demographic group selected.")

        clustering = shap.utils.hclust(X, self.y_raw)
        shap.plots.bar(shap_vals, clustering=clustering, clustering_cutoff=0.5)


