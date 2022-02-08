"""File containing feature selection classes."""
import sys
import warnings
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn_relief as relief
import xgboost as xgb
from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from biondeep_ig.src import FEATURES_DIRECTORY
from biondeep_ig.src import FS_CONFIGURATION_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.utils import get_task_name

log = get_logger("FeatureSelection/FS")
warnings.filterwarnings("ignore")
original_stdout = sys.__stdout__

Path(FS_CONFIGURATION_DIRECTORY / "FS_feature_lists_created").mkdir(parents=True, exist_ok=True)


# ---------------------------------------- #
class BaseFeatureSelection(ABC):
    """Basic class Method."""

    def __init__(
        self,
        features,
        force_features,
        label_name,
        other_params,
        folder_name,
        n_feat,
        fs_type,
    ):
        """Init method."""
        self.features = features
        self.force_features = force_features
        self.label_name = label_name
        self.other_params = other_params
        self.folder_name = folder_name
        self.n_feat = n_feat
        self.fs_type = fs_type
        self.fs = None

    @property
    def fs_meta_data(self):
        """Fs meta data."""
        return {
            "features": self.features,
            "force_features": self.force_features,
        }

    @abstractmethod
    def rank_features(self, df_processed, targets_c):
        """Rank features method."""

    def select_features(self, df_processed, label_s):
        """Select n features method."""
        # get current feature subfolder
        task = get_task_name(self.label_name)
        df_stats = self.rank_features(df_processed, label_s)

        df_stats_sort = df_stats.sort_values(by="Rank", ascending=[False])

        feature_list = df_stats_sort.iloc[: self.n_feat]
        feature_list = list(feature_list.index.tolist())
        log.info(feature_list)
        if self.force_features:
            feature_list.extend(self.force_features)
            feature_list = list(set(feature_list))

        with open(
            FEATURES_DIRECTORY / task / f"{self.folder_name}_{self.fs_type}.txt", "w+"
        ) as outfile:
            for item in feature_list:
                outfile.write(f"{item}\n")


# ---------------------------------------- #
class Fspca(BaseFeatureSelection):
    """PCA feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        pca = PCA()  # n_components = x
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        _ = pca.fit_transform(x)
        n_pcs = pca.components_.shape[0]
        initial_feature_names = df_processed.columns.to_list()
        pca_comps_sorted = [-np.sort(-np.abs(pca.components_[i, :])) for i in range(n_pcs)]
        importance_feature = []
        for j in range(len(pca_comps_sorted[0])):
            importance_feature.append(
                [
                    np.where(np.abs(pca.components_[i]) == pca_comps_sorted[i][j])[0].tolist()[0]
                    for i in range(n_pcs)
                ]
            )
        most_important_names = []
        for j in range(len(pca_comps_sorted[0])):
            for k in range(self.other_params["SamplesPerPCA"]):
                most_important_names.append(initial_feature_names[importance_feature[k][j]])
        most_important_names = list(dict.fromkeys(most_important_names))
        ranki = len(most_important_names)

        dict_stats = {}
        for item in most_important_names:
            dict_stats[item] = ranki
            ranki -= 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsxgboost(BaseFeatureSelection):
    """xgboost feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        targets_c = targets_c.to_numpy().astype(int)
        model = xgb.XGBClassifier(
            max_depth=self.other_params["max_depth"],
            learning_rate=self.other_params["learning_rate"],
            nthread=self.other_params["nthread"],
            random_state=self.other_params["rand_seed"],
        )
        bst = model.fit(x, targets_c.ravel())
        bst.get_booster().feature_names = df_processed.loc[
            :, df_processed.columns != self.label_name
        ].columns.tolist()
        feat_imp = bst.get_booster().get_score(importance_type="weight")
        dict_stats = {}
        for key, value in feat_imp.items():
            dict_stats[key] = value
        plt.figure()
        xgb.plot_importance(bst)
        plt.tight_layout()
        plt.savefig(
            FS_CONFIGURATION_DIRECTORY
            / "FS_feature_lists_created"
            / "FeatureSelection_XGBoost_Importance.pdf"
        )
        plt.clf()
        plt.close()

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsxgboostshap(BaseFeatureSelection):
    """xgboost shap feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        cols = df_processed.columns
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        targets_c = targets_c.to_numpy().astype(int)
        model = xgb.XGBClassifier(
            max_depth=self.other_params["max_depth"],
            learning_rate=self.other_params["learning_rate"],
            nthread=self.other_params["nthread"],
            random_state=self.other_params["rand_seed"],
        )
        bst = model.fit(x, targets_c.ravel())
        bst.get_booster().feature_names = df_processed.loc[
            :, df_processed.columns != self.label_name
        ].columns.tolist()
        shap_values = shap.TreeExplainer(bst).shap_values(x)
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        dict_stats = {}
        for key, value in zip(cols, mean_shap_values):
            dict_stats[key] = value
        plt.figure()
        shap.summary_plot(shap_values, x, feature_names=cols, show=False)
        plt.tight_layout()
        plt.savefig(
            FS_CONFIGURATION_DIRECTORY
            / "FS_feature_lists_created"
            / "FeatureSelection_xGBoost_Shap.pdf"
        )
        plt.clf()
        plt.close()

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsrfgini(BaseFeatureSelection):
    """Random forest gini feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        targets_c = targets_c.to_numpy().astype(int)
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        interpolator = RandomForestClassifier(
            n_estimators=self.other_params["n_estimators"],
            max_features=self.other_params["max_features"],
            max_depth=self.other_params["max_depth"],
            random_state=self.other_params["rand_seed"],
            criterion=self.other_params["criterion"],
            n_jobs=self.other_params["nthread"],
        )
        interpolator = interpolator.fit(x, targets_c.ravel())
        interpolator.feature_names = df_processed.loc[
            :, df_processed.columns != self.label_name
        ].columns.tolist()
        features = df_processed.loc[:, df_processed.columns != self.label_name].columns.tolist()
        importances = interpolator.feature_importances_
        indices = np.argsort(importances)
        dict_stats = {}
        for key, value in zip(features, interpolator.feature_importances_):
            dict_stats[key] = value
        plt.figure()
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(
            FS_CONFIGURATION_DIRECTORY
            / "FS_feature_lists_created"
            / "FeatureSelection_RFgini_Importance.pdf"
        )
        plt.clf()
        plt.close()

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fscossim(BaseFeatureSelection):
    """Cosine similarity feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        cs = cosine_similarity(x.T, Y=None, dense_output=True)
        plt.figure()
        a = plt.pcolor(cs)
        plt.colorbar(a)
        plt.xticks(
            np.arange(np.shape(cs)[0]),
            df_processed.loc[:, df_processed.columns != self.label_name].columns.tolist(),
            rotation="vertical",
        )
        plt.yticks(
            np.arange(np.shape(cs)[0]),
            df_processed.loc[:, df_processed.columns != self.label_name].columns.tolist(),
            rotation="horizontal",
        )
        plt.tight_layout()
        plt.savefig(
            FS_CONFIGURATION_DIRECTORY
            / "FS_feature_lists_created"
            / "FeatureSelection_CosSimilarity.pdf"
        )
        plt.clf()
        plt.close()
        ranks = 1 / np.max(cs, axis=0)
        i = 0
        dict_stats = {}
        for col in df_processed.loc[:, df_processed.columns != self.label_name].columns.tolist():
            dict_stats[col] = ranks[i]
            i += 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsrfelr(BaseFeatureSelection):
    """Random feature elemination based on logistic regression feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        lr = LogisticRegression(fit_intercept=False, n_jobs=self.other_params["n_thread"])
        lr.fit(x, targets_c)
        fs = RFE(lr, n_features_to_select=self.n_feat)
        fs = fs.fit(x, targets_c)
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[fs.support_]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fssfmlr(BaseFeatureSelection):
    """select from model based on logistic regression feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        lr = LogisticRegression(fit_intercept=False, n_jobs=self.other_params["n_thread"])
        lr.fit(x, targets_c)
        fs = SelectFromModel(lr, threshold=-np.inf, max_features=self.n_feat)
        fs = fs.fit(x, targets_c)
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[fs.get_support()]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsrfexgboost(BaseFeatureSelection):
    """Random feature elemination based on xgboost feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        lr = xgb.XGBClassifier(
            max_depth=self.other_params["max_depth"],
            learning_rate=self.other_params["learning_rate"],
            nthread=self.other_params["nthread"],
            random_state=self.other_params["rand_seed"],
        )
        lr.fit(x, targets_c)
        fs = RFE(lr, n_features_to_select=self.n_feat)
        fs = fs.fit(x, targets_c)
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[fs.support_]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fssfmxgboost(BaseFeatureSelection):
    """select from model based on xgboost feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        lr = xgb.XGBClassifier(
            max_depth=self.other_params["max_depth"],
            learning_rate=self.other_params["learning_rate"],
            nthread=self.other_params["nthread"],
            random_state=self.other_params["rand_seed"],
        )
        lr.fit(x, targets_c)
        fs = SelectFromModel(lr, threshold=-np.inf, max_features=self.n_feat)
        fs = fs.fit(x, targets_c)
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[fs.get_support()]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsboruta(BaseFeatureSelection):
    """Boruta feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        log.info(
            "UserInfo: This method does not take into account the forced number of features ..."
        )
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        forest = RandomForestClassifier(
            n_estimators=self.other_params["n_estimators"],
            max_features=self.other_params["max_features"],
            max_depth=self.other_params["max_depth"],
            random_state=self.other_params["rand_seed"],
            criterion=self.other_params["criterion"],
            n_jobs=self.other_params["nthread"],
        )
        boruta = BorutaPy(
            estimator=forest,
            n_estimators=self.other_params["n_estimators"],
            max_iter=self.other_params["max_iter"],
            random_state=self.other_params["rand_seed"],
            verbose=0,
        )
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        targets_c = targets_c.to_numpy().astype(int).ravel()
        boruta.fit(x, targets_c)
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[boruta.support_]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsrelief(BaseFeatureSelection):
    """Relief feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        targets_c = targets_c.to_numpy().astype(int).ravel()
        r = relief.Relief(
            n_features=self.n_feat,
            n_jobs=self.other_params["n_threads"],
            random_state=self.other_params["rand_seed"],
        )
        _ = r.fit_transform(x, targets_c)
        i = 0
        dict_stats = {}
        for col in df_processed.loc[:, df_processed.columns != self.label_name].columns.tolist():
            dict_stats[col] = r.w_[i]
            i += 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fsmi(BaseFeatureSelection):
    """Mutual information feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        df_processed, targets_c = dropnans(
            df_processed, targets_c
        )  # this one needs NaNs to be dropped
        x = df_processed.loc[:, df_processed.columns != self.label_name].to_numpy()
        targets_c = targets_c.to_numpy().astype(int).ravel()
        selection = SelectKBest(mutual_info_classif, k=self.n_feat).fit(
            x, targets_c
        )  # gives back a boolean matrix
        dict_stats = {}
        for col in (
            df_processed.loc[:, df_processed.columns != self.label_name]
            .columns[selection.get_support()]
            .tolist()
        ):
            dict_stats[col] = 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


# ---------------------------------------- #
class Fscorr(BaseFeatureSelection):
    """Correlation between features and target feature selection class."""

    def rank_features(self, df_processed, targets_c):
        """Run the feature selection method."""
        f = plt.figure(figsize=(19, 15))
        df_processed[self.label_name] = targets_c.astype(int)
        corrmat = df_processed.select_dtypes(["number"]).corr()
        plt.matshow(corrmat, fignum=f.number, cmap=matplotlib.cm.Spectral_r)
        plt.xticks(
            range(df_processed.select_dtypes(["number"]).shape[1]),
            df_processed.select_dtypes(["number"]).columns,
            rotation=90,
        )
        plt.yticks(
            range(df_processed.select_dtypes(["number"]).shape[1]),
            df_processed.select_dtypes(["number"]).columns,
        )
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=12)
        plt.title("Correlation Matrix", fontsize=12)
        plt.savefig(
            FS_CONFIGURATION_DIRECTORY / "FS_feature_lists_created" / "FeatureSelection_corrmat.pdf"
        )
        plt.clf()
        plt.close()

        featnames = corrmat.columns.to_list()
        featnames = [x for x in featnames if x != self.label_name]
        corrwithtarget = np.abs(corrmat[self.label_name].to_numpy())
        corrwithtarget = corrwithtarget[np.where(corrwithtarget < 1.0)]  # remove corr with itself

        # Create ranks based on high correlation
        i = 0
        dict_stats = {}
        for col, val in zip(featnames, corrwithtarget):
            dict_stats[col] = val
            i += 1

        return pd.DataFrame.from_dict(dict_stats, orient="index", columns=["Rank"])


def dropnans(df, target):
    """Droping the NaN values."""
    inds = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    df = df.drop(inds)
    target = target.drop(inds)
    inds = pd.isnull(target).any().nonzero()[0]
    df = df.drop(inds)
    target = target.drop(inds)
    return df, target
