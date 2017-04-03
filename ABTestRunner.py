import numpy as np
from scipy import stats

class ABTestRunner(object):
    """Some ways to do AB tests"""

    def __init__(self, name, alpha, alternative="<>"):
        """Define the test name

        Args:
                name (String): the test name
                alpha (Double): significance level
                alternative hypothesis (String): Whether uB (>, <, <>) uA
        """

        if name not in ["plain_t_test",
                        "unbalanced_groups_t_test",
                        "welchs_t_test",
                        "kolmogorov_smirnov",
                        "mann_whitneyu",
                        "kruskal_wallis",
                        "binary_chi2"]:
            raise KeyError("The test is not implemented yet")

        if alternative not in [">", "<", "<>"]:
            raise KeyError("The alternate hypothesis can be either >, < or <>")

        if alpha <= 0.0:
            raise ValueError("The significance level must be positive")

        self.alpha = alpha
        self.name = name
        self.alternative = alternative

        if name == "plain_t_test":
            self.test = plain_t_test
        elif name == "unbalanced_groups_t_test":
            self.test = unbalanced_groups_t_test
        elif name == "welchs_t_test":
            self.test = welchs_t_test
        elif name == "kolmogorov_smirnov":
            self.test = kolmogorov_smirnov
        elif name == "mann_whitneyu":
            self.test = mann_whitneyu
        elif name == "kruskal_wallis":
            self.test = kruskal_wallis
        elif name == "binary_chi2":
            self.test = binary_chi2

    def run_test(self, data_samples):
        """Run the test

        Args:
                data_samples (List[numpy.array]): each element of the list
                        is an array of observations
        """
        assert isinstance(data_samples, list)
        for el in data_samples:
            assert type(el) == np.ndarray

        pval = self.test(data_samples, self.alternative)
        print("[{} on uA {} uB]\tP={} -> Ho {} rejected".format(
            self.name,
            self.alternative,
            pval,
            "IS" if pval < self.alpha else "CANNOT BE"))


def plain_t_test(data, alternative):
    """plain t-test, for 2 groups of samples with same length.
    Try `unbalanced_groups_t_test` if the samples have different lengths.

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * The distribution is assumed gaussian in both groups
        * The variance is assumed to be the same
    """

    if len(data) != 2:
        raise ValueError("2 groups are needed")

    a = data[0]
    b = data[1]
    N = a.shape[0]

    if a.shape != b.shape:
        raise ValueError("The 2 groups must have the same number of observations")

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    sp = np.sqrt((var_a + var_b)/2.0)
    t = (np.mean(a) - np.mean(b)) / (sp * np.sqrt(2.0/N))
    df = 2*N - 2

    if alternative == "<>":
        p = stats.t.cdf(np.fabs(t)*-1, df=df) * 2
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, p2))
    elif alternative == ">":
        # t should be negative
        p = 1.0 - stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, p2))
    elif alternative == "<":
        # t should be positive
        p = stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, 1-p2))

    return p


def unbalanced_groups_t_test(data, alternative):
    """t-test, for 2 groups of samples with different length.
    If they have the same length, you can use also plain_t_test

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * The distribution is assumed gaussian in both groups
        * The variance is assumed to be the same
    """

    if len(data) != 2:
        raise ValueError("2 groups are needed")

    a = data[0]
    b = data[1]
    n_a = a.shape[0]
    n_b = b.shape[0]

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    sp = np.sqrt(((n_a-1)*var_a + (n_b-1)*var_b)/(n_a+n_b-2))
    t = (np.mean(a) - np.mean(b)) / (sp * np.sqrt(1./n_a + 1./n_b))
    df = n_a + n_b - 2

    if alternative == "<>":
        p = stats.t.cdf(np.fabs(t)*-1, df=df) * 2
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, p2))
    elif alternative == ">":
        # t should be negative
        p = 1.0 - stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, p2))
    elif alternative == "<":
        # t should be positive
        p = stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b)
        assert(np.isclose(p, 1-p2))
    return p


def welchs_t_test(data, alternative):
    """Welch's t-test, for 2 groups of samples with any length and variance

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * It's assumed the two groups have normal distribution

    """

    if len(data) != 2:
        raise ValueError("2 groups are needed")

    a = data[0]
    b = data[1]

    n_a = a.shape[0]
    n_b = b.shape[0]

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    nu_a = n_a - 1
    nu_b = n_b - 1

    sd = np.sqrt(var_a/n_a + var_b/n_b)
    t = (np.mean(a) - np.mean(b)) / sd

    df = (sd ** 4) / ((var_a ** 2)/(nu_a * (n_a ** 2)) + (var_b ** 2)/(nu_b * (n_b ** 2)))

    if alternative == "<>":
        p = stats.t.cdf(np.fabs(t)*-1, df=df) * 2
        _, p2 = stats.ttest_ind(a, b, axis=0, equal_var=False)
        assert(np.isclose(p, p2))
    elif alternative == ">":
        # t should be negative
        p = 1.0 - stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b, axis=0, equal_var=False)
        assert(np.isclose(p, p2))
    elif alternative == "<":
        # t should be positive
        p = stats.t.cdf(t, df=df)
        _, p2 = stats.ttest_ind(a, b, axis=0, equal_var=False)
        assert(np.isclose(p, 1-p2))
    return p


def kolmogorov_smirnov(data, alternative):
    """Kolmogorov-Smirnow test, for 2 groups of samples with any length and variance.
    It tests the distance between the empirical distribution function of the two groups.

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * It's a non-parametric test

    """

    if len(data) != 2:
        raise ValueError("2 groups are needed")

    if alternative != "<>":
        raise ValueError("KS doesn't work on 1-sided problems")

    a = data[0]
    b = data[1]

    _, p = stats.ks_2samp(a, b)
    return p


def mann_whitneyu(data, alternative):
    """Mann Whitneyu 's U-test, for 2 groups of samples with any length. It tests whether
    the two groups come from the same population, any distributed.

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * It's a non-parametric test

    """

    if len(data) != 2:
        raise ValueError("2 groups are needed")

    a = data[0]
    b = data[1]

    if alternative == "<>":
        _, p = stats.mannwhitneyu(a, b)
        p *= 2
    elif alternative == ">":
        _, p = stats.mannwhitneyu(a, b)
    elif alternative == "<":
        _, p = stats.mannwhitneyu(a, b)
        p = 1-p
    return p


def kruskal_wallis(data, alternative):
    """Kruskal-Wallis 's H-test, for K groups of samples with any length and variance. It tests whether
    the two groups come from the same population, any distributed.

    Args:
        data (List[numpy.array]): each element of the list is an array of observations
        alternative (String): Whether uA, should be >, < or <> than uB

    Note:
        * It's a non-parametric test
        * It's an extension of the mann_whitneyu U-test for any number of group.
    """

    if alternative == "<>":
        _, p = stats.mannwhitneyu(*data)
        p *= 2
    elif alternative == ">":
        _, p = stats.mannwhitneyu(*data)
    elif alternative == "<":
        _, p = stats.mannwhitneyu(*data)
        p = 1-p
    return p


def binary_chi2(data, alternative):
    """
    """

    for group in data:
        assert issubclass(group.dtype.type, np.integer)
        assert set(np.unique(group)) == set((0, 1))

    n_groups = len(data)
    n_outcomes = 2
    ct = np.zeros((n_groups, n_outcomes))

    for i, group in enumerate(data):
        for el in group:
            ct[i, el] += 1

    support_outcome = np.sum(ct, axis=0)
    support_group = np.sum(ct, axis=1)
    n_samples = np.sum(ct, axis=None)


    chi2 = 0.0
    for i in range(n_groups):
        for j in range(n_outcomes):
            observed = ct[i, j]
            expected = support_group[i] * support_outcome[j] / n_samples
            chi2 += ((observed-expected) ** 2) / expected

    _, p, _, _ = stats.chi2_contingency(ct, correction=False)
    return p

if __name__ == "__main__":
    import sys
    from GenerateSamples import RealGroupParams, BinaryGroupParams, GenerateRealSamples, GenerateBinarySamples

    real_group_specs = [RealGroupParams(20, 4, 15),
                        RealGroupParams(22, 4, 15)]

    data = [GenerateRealSamples(real_group_specs).get_samples_for_group(i) for i in range(len(real_group_specs))]


    # Datasets taken from https://en.wikipedia.org/wiki/Welch%27s_t-test
    # A=N(20, 4), B=N(22,4)
    data1 = [
        np.array([27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4]),
        np.array([27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4]),
    ]

    # A=N(20, 16), B=N(22, 1) + unequal group size
    data2 = [
        np.array([17.2, 20.9, 22.6, 18.1, 21.7, 21.4, 23.5, 24.2, 14.7, 21.8]),
        np.array([21.5, 22.8, 21.0, 23.0, 21.6, 23.6, 22.5, 20.7, 23.4, 21.8, 20.7, 21.7, 21.5, 22.5, 23.6, 21.5, 22.5, 23.5, 21.5, 21.8]),
    ]

    # A=N(20, 1), B=N(22, 16) + unequal group size
    data3 = [
        np.array([19.8, 20.4, 19.6, 17.8, 18.5, 18.9, 18.3, 18.9, 19.5, 22.0]),
        np.array([28.2, 26.6, 20.1, 23.3, 25.2, 22.1, 17.7, 27.6, 20.6, 13.7, 23.2, 17.5, 20.6, 18.0, 23.9, 21.6, 24.3, 20.4, 24.0, 13.2]),
    ]

    # From https://en.wikipedia.org/wiki/Student%27s_t-test
    data4 = [
        np.array([30.02, 29.99, 30.11, 29.97, 30.01, 29.99]),
        np.array([29.89, 29.93, 29.72, 29.98, 30.02, 29.98])
    ]

    for i, data in enumerate([data, data1, data2, data3, data4]):
        print("dataset # {}".format(i+1))
        for test_name in ["plain_t_test", "unbalanced_groups_t_test",
                          "welchs_t_test", "kolmogorov_smirnov",
                          "mann_whitneyu", "kruskal_wallis"]:
            try:
                tests = ABTestRunner(name=test_name, alpha=0.05)
                tests.run_test(data)
            except Exception as e:
                print("[{}]: Error: {}".format(test_name, sys.exc_info()[0]))


    binary_group_specs = [BinaryGroupParams(0.72, 50),
                          BinaryGroupParams(30/55.0+0.001, 55)]

    bin_data = [GenerateBinarySamples(binary_group_specs).get_samples_for_group(i) for i in range(len(binary_group_specs))]


    tests = ABTestRunner(name="binary_chi2", alpha=0.05)
    tests.run_test(bin_data)