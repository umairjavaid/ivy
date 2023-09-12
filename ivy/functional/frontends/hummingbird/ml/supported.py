def _build_sklearn_operator_list():
    """
    Put all supported Sklearn operators on a list.
    """
    if sklearn_installed():
        # Tree-based models
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            IsolationForest,
            RandomForestClassifier,
            RandomForestRegressor,
        )

        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Linear-based models
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            LogisticRegressionCV,
            SGDClassifier,
            RidgeCV,
            ElasticNet,
            Ridge,
            Lasso,
            TweedieRegressor,
            PoissonRegressor,
            GammaRegressor,
        )

        # SVM-based models
        from sklearn.svm import LinearSVC, SVC, NuSVC, LinearSVR

        # Imputers
        from sklearn.impute import MissingIndicator, SimpleImputer

        # MLP Models
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        # Naive Bayes Models
        from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

        # Matrix decomposition transformers
        from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD

        # Cross decomposition
        from sklearn.cross_decomposition import PLSRegression

        # KNeighbors models
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neighbors import KNeighborsRegressor

        # Clustering models
        from sklearn.cluster import KMeans, MeanShift

        # Model selection
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        # Preprocessing
        from sklearn.preprocessing import (
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            OneHotEncoder,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,
        )

        try:
            from sklearn.preprocessing import Imputer
        except ImportError:
            # Imputer was deprecate in sklearn >= 0.22
            Imputer = None

        # Features
        from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

        # Mixture models
        from sklearn.mixture import BayesianGaussianMixture

        supported_ops = [
            # Trees
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            IsolationForest,
            OneHotEncoder,
            RandomForestClassifier,
            RandomForestRegressor,
            # Linear-methods
            LinearRegression,
            LinearSVC,
            LinearSVR,
            LogisticRegression,
            LogisticRegressionCV,
            SGDClassifier,
            RidgeCV,
            Lasso,
            ElasticNet,
            Ridge,
            TweedieRegressor,
            PoissonRegressor,
            GammaRegressor,
            # Clustering
            KMeans,
            MeanShift,
            # Other models
            BernoulliNB,
            GaussianNB,
            KNeighborsClassifier,
            KNeighborsRegressor,
            MLPClassifier,
            MLPRegressor,
            MultinomialNB,
            # SVM
            NuSVC,
            SVC,
            # Imputers
            Imputer,
            MissingIndicator,
            SimpleImputer,
            # Preprocessing
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,
            # Matrix Decomposition
            FastICA,
            KernelPCA,
            PCA,
            TruncatedSVD,
            # Cross Decomposition
            PLSRegression,
            # Feature selection
            SelectKBest,
            SelectPercentile,
            VarianceThreshold,
            # Mixture models
            BayesianGaussianMixture,
        ]

        # Remove all deprecated operators given the sklearn version. E.g., Imputer for sklearn > 0.21.3.
        return [x for x in supported_ops if x is not None]

    return []

sklearn_operator_list = _build_sklearn_operator_list()

def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    # Pipeline ops. These are ops injected by the parser not "real" sklearn operators.
    pipeline_operator_list = [
        "ArrayFeatureExtractor",
        "Concat",
        "Multiply",
        "Bagging",
    ]

    return {
        k: "Sklearn" + k.__name__ if hasattr(k, "__name__") else k
        for k in sklearn_operator_list
        + pipeline_operator_list
        + xgb_operator_list
        + lgbm_operator_list
        + prophet_operator_list
    }

sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()

def get_sklearn_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *scikit-learn API* format.

    Args:
        model_type: A scikit-learn model object (e.g., RandomForestClassifier)
                    or an object with scikit-learn API (e.g., LightGBM)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework
    """
    if model_type not in sklearn_api_operator_name_map:
        raise MissingConverter("Unable to find converter for model type {}.".format(model_type))
    return sklearn_api_operator_name_map[model_type]