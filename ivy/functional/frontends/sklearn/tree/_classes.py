from abc import ABCMeta, abstractmethod
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
)

<<<<<<< HEAD
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):

=======

class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
>>>>>>> upstream/main
    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def get_depth(self):
        raise NotImplementedError
<<<<<<< HEAD
        
=======
>>>>>>> upstream/main

    def get_n_leaves(self):
        raise NotImplementedError

<<<<<<< HEAD

    def _support_missing_values(self, X):
        raise NotImplementedError


    def _compute_missing_values_in_feature_mask(self, X):
        raise NotImplementedError


=======
    def _support_missing_values(self, X):
        raise NotImplementedError

    def _compute_missing_values_in_feature_mask(self, X):
        raise NotImplementedError

>>>>>>> upstream/main
    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        raise NotImplementedError

<<<<<<< HEAD

    def _validate_X_predict(self, X, check_input):
        raise NotImplementedError


    def predict(self, X, check_input=True):
        raise NotImplementedError
    
    def apply(self, X, check_input=True):
        raise NotImplementedError
    
=======
    def _validate_X_predict(self, X, check_input):
        raise NotImplementedError

    def predict(self, X, check_input=True):
        raise NotImplementedError

    def apply(self, X, check_input=True):
        raise NotImplementedError
>>>>>>> upstream/main

    def decision_path(self, X, check_input=True):
        raise NotImplementedError

<<<<<<< HEAD

    def _prune_tree(self):
        raise NotImplementedError


    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        raise NotImplementedError


=======
    def _prune_tree(self):
        raise NotImplementedError

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        raise NotImplementedError

>>>>>>> upstream/main
    @property
    def feature_importances_(self):
        raise NotImplementedError


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
<<<<<<< HEAD

=======
>>>>>>> upstream/main
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def fit(self, X, y, sample_weight=None, check_input=True):
<<<<<<< HEAD

=======
>>>>>>> upstream/main
        super()._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        return self

    def predict_proba(self, X, check_input=True):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def _more_tags(self):
        allow_nan = self.splitter == "best" and self.criterion in {
            "gini",
            "log_loss",
            "entropy",
        }
<<<<<<< HEAD
        return {"multilabel": True, "allow_nan": allow_nan}
=======
        return {"multilabel": True, "allow_nan": allow_nan}
>>>>>>> upstream/main
