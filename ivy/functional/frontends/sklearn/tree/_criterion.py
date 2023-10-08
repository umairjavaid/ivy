import ivy


EPSILON = 10 * ivy.finfo("float64").eps


class Criterion:
    """
    Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is
    using different metrics.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def init(
        self,
        y,
        sample_weight,
        weighted_n_samples: float,
        sample_indices: list,
        start: int,
        end: int,
    ):
        """
        Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        pass

    def init_missing(self, n_missing):
        """
        Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: SIZE_t
            Number of missing values for specific feature.
        """
        pass

    def reset(self):
        """
        Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    def reverse_reset(self):
        """
        Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    def update(self, new_pos):
        """
        Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the sample_indices in the right child
        """
        pass

    def node_impurity(self):
        """
        Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of the
        current node, i.e. the impurity of sample_indices[start:end].
        This is the primary function of the criterion class. The smaller
        the impurity the better.
        """
        pass

    def children_impurity(self, impurity_left, impurity_right):
        """
        Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """
        return impurity_left, impurity_right

    def node_value(self, dest):
        """
        Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """
        return dest

    def proxy_impurity_improvement(self):
        """
        Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this
        value also maximizes the impurity improvement. It neglects all
        constant terms of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        impurity_left = 0.0
        impurity_right = 0.0
        impurity_left, impurity_right = self.children_impurity(
            impurity_left, impurity_right
        )

        return (
            -self.weighted_n_right * impurity_right
            - self.weighted_n_left * impurity_left
        )

    def impurity_improvement(
        self, impurity_parent: float, impurity_left: float, impurity_right: float
    ):
        """
        Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child.

        Parameters
        ----------
        impurity_parent : double
            The initial impurity of the parent node before the split

        impurity_left : double
            The impurity of the left child

        impurity_right : double
            The impurity of the right child

        Return
        ------
        double : improvement in impurity after the split occurs
        """
        return (self.weighted_n_node_samples / self.weighted_n_samples) * (
            impurity_parent
            - (self.weighted_n_right / self.weighted_n_node_samples) * impurity_right
            - (self.weighted_n_left / self.weighted_n_node_samples) * impurity_left
        )

    def init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        pass


class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __init__(self, n_outputs: int, n_classes: ivy.Array):
        """
        Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
        self.start = 0
        self.pos = 0
        self.end = 0
        self.missing_go_to_left = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.n_classes = ivy.empty(n_outputs, dtype=ivy.int16)

        k = 0
        max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        # Count labels for each output
        self.sum_total = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_left = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)
        self.sum_right = ivy.zeros((n_outputs, max_n_classes), dtype=ivy.float64)

    def __reduce__(self):
        return (
            type(self),
            (self.n_outputs, ivy.asarray(self.n_classes)),
            self.__getstate__(),
        )

    def init(
        self,
        y: list,
        sample_weight: list,
        weighted_n_samples: float,
        sample_indices: int,
        start: int,
        end: int,
    ):
        """
        Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of all samples
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        i = 0
        p = 0
        k = 0
        c = 0
        w = 1.0

        for k in range(self.n_outputs):
            self.sum_total[k] = 0

        for p in range(start, end):
            # print(f"{p=}")
            i = sample_indices[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_total[k, c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    def init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = ivy.zeros(
            (self.n_outputs, self.max_n_classes), dtype=ivy.float64
        )

    def init_missing(self, n_missing):
        """
        Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        #print(f"init_missing: {n_missing=}")
        #input()
        i = 0
        p = 0
        k = 0
        y_ik = 0
        c = 0
        w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        self.sum_missing[0, 0 : self.max_n_classes * self.n_outputs * 8] = 0
        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                c = int(self.y[i, k])
                self.sum_missing[k, c] += w

            self.weighted_n_missing += w

    def reset(self):
        """
        Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        self.pos = self.start
        self.weighted_n_left, self.weighted_n_right = _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            self.weighted_n_left,
            self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    def reverse_reset(self):
        """
        Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        self.pos = self.end
        self.weighted_n_right, self.weighted_n_left = _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            self.weighted_n_right,
            self.weighted_n_left,
            not self.missing_go_to_left,
        )
        return 0

    def update(self, new_pos):
        """
        Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        pos = self.pos
        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        end_non_missing = self.end - self.n_missing

        sample_indices = self.sample_indices
        sample_weight = self.sample_weight

        i = 0
        p = 0
        k = 0
        c = 0
        w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    c = int(self.y[i, k])
                    self.sum_left[k, c] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    c = int(self.y[i, k])
                    self.sum_left[k, c] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left

        for k in range(self.n_outputs):
            for c in range(self.max_n_classes):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    def node_impurity(self):
        pass

    def children_impurity(self, impurity_left: float, impurity_right: float):
        pass

    def node_value(self, dest):
        """
        Compute the node value of sample_indices[start:end] and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        print("---node_value---")
        print(f"dest: {dest}")
        print(f"self.n_outputs: {self.n_outputs}")
        print(f"self.sum_total: {self.sum_total}")
        print(f"self.n_classes: {self.n_classes}")
        print("---node_value---")
        #TODO: THIS IS NOT THE CORRECT IMPLEMENTATION. CORRECT THIS
        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k, 0].copy()

        return dest



class Gini(ClassificationCriterion):
    """
    Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """
    #TODO: can we do this with a simple matrix multiplication?
    def node_impurity(self):
        """
        Evaluate the impurity of the current node.

        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the
        impurity the better.
        """
        gini = 0.0
        sq_count = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(int(self.n_classes[k])):
                count_k = self.sum_total[k, c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (
                self.weighted_n_node_samples * self.weighted_n_node_samples
            )

        return gini / self.n_outputs

    def children_impurity(
        self,
        impurity_left: float,
        impurity_right: float,
    ):
        """
        Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node to
        impurity_right : double pointer
            The memory address to save the impurity of the right node to
        """
        gini_left = 0.0
        gini_right = 0.0
        sq_count_left = 0.0
        sq_count_right = 0.0
        count_k = 0.0
        k = 0
        c = 0

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            #TODO: check if the following implementation is correct, added print
            print(f"self.n_classes: {self.n_classes}")
            #TODO: I added the following int because I was getting the TypeError: 'Array' object cannot be interpreted as an integer
            for c in range(int(self.n_classes[k])):
                count_k = self.sum_left[k, c]
                sq_count_left += count_k * count_k

                count_k = self.sum_right[k, c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (
                self.weighted_n_left * self.weighted_n_left
            )
            gini_right += 1.0 - sq_count_right / (
                self.weighted_n_right * self.weighted_n_right
            )

        impurity_left = gini_left / self.n_outputs
        impurity_right = gini_right / self.n_outputs

        return impurity_left, impurity_right


# --- Helpers --- #
# --------------- #

#TODO: discuss this with illia, ptr allocations. Are these needed? do we need to allocate memory here
def _move_sums_classification(
    criterion, sum_1, sum_2, weighted_n_1, weighted_n_2, put_missing_in_1
):
    """
    Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values go to sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    print("---_move_sums_classification---")
    print("forced an if condition. Fix this according to the original implementation")
    print("---_move_sums_classification---")
    # if criterion.n_missing != 0 and put_missing_in_1:
    #TODO: added the following, wasnt running with GINI
    #if put_missing_in_1:
    if False:
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] 
            sum_1[k, 0:n_bytes] = criterion.sum_missing[k, 0:n_bytes]

        for k in range(criterion.n_outputs):
            for c in range(criterion.n_classes[k]):
                sum_2[k, c] = criterion.sum_total[k, c] - criterion.sum_missing[k, c]

        weighted_n_1 = criterion.weighted_n_missing
        weighted_n_2 = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        # Assigning sum_2 = sum_total for all outputs.
        for k in range(criterion.n_outputs):
            n_bytes = int(criterion.n_classes[k])
            sum_1[k, 0:n_bytes] = 0
            sum_2[k, 0:n_bytes] = criterion.sum_total[k, 0:n_bytes]

        weighted_n_1 = 0.0
        weighted_n_2 = criterion.weighted_n_node_samples

    return weighted_n_1, weighted_n_2
