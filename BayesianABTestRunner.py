import numpy as np


class BayesianABTestRunner(object):
    """Epsilon greedy approach to run tests"""

    def __init__(self, name):
        """Define the test name

        Args:
                name (String): the test name

        Note:
            * all tests are 2-sided
        """

        if name not in ["ucb1",
                        "bandits"]:
            raise KeyError("The test is not implemented yet")

        if name == "ucb1":
            self.test = ucb1
        if name == "bandits":
            self.test = bandits

    def run_test(self, data_samples, n_steps, epsilon=None):
        """Run the test

        Args:
            data_samples(List[numpy.array]): each element of the list is an array of observations
            n_steps: the number of steps to run
            epsilon (Double): The exploration epsilon (if needed by the algo)
        """
        assert isinstance(data_samples, list)
        for el in data_samples:
            assert type(el) == np.ndarray

        assert n_steps > 0
        for group in data_samples:
            assert issubclass(group.dtype.type, np.integer)
            assert set(np.unique(group)) == set((0, 1))

        self.test(data_samples, n_steps, epsilon)


def ucb1(data_samples, n_steps, epsilon):
    """
    Epsilon greedy, or UCB1 algorithm
    
    """
    n_groups = len(data_samples)
    punctual_means = np.zeros((n_groups, ))
    punctual_extractions = [list([]) for _ in range(n_groups)]

    idxs = np.zeros((n_groups, ), dtype=int)

    extractions = []

    for i in range(n_steps):
        r = np.random.rand()

        if r < epsilon:
            # Explore
            arm = np.random.randint(n_groups)
        else:
            # Exploit
            arm = np.argmax(punctual_means)

        extraction = (data_samples[arm])[idxs[arm]]
        idxs[arm] += 1

        extractions.append(extraction)
        punctual_extractions[arm].append(extraction)
        punctual_means[arm] = np.mean(punctual_extractions[arm])

        if i % 100 == 0:
            print("step", i,
                  "global mean", np.mean(extractions),
                  "arm performance", punctual_means)


def bandits(data_samples, n_steps, *args):
    """
    Multi arm bandits

    """
    n_groups = len(data_samples)

    alpha_beta_params = [{'a': 1.0, 'b': 1.0} for _ in range(n_groups)]

    punctual_means = np.zeros((n_groups,))
    punctual_extractions = [list([]) for _ in range(n_groups)]
    idxs = np.zeros((n_groups,), dtype=int)

    extractions = []

    for i in range(n_steps):
        samples = [np.random.beta(bandit['a'], bandit['b']) for bandit in alpha_beta_params]
        arm = np.argmax(samples)

        extraction = (data_samples[arm])[idxs[arm]]
        idxs[arm] += 1

        alpha_beta_params[arm]['a'] += extraction
        alpha_beta_params[arm]['b'] += 1 - extraction

        extractions.append(extraction)
        punctual_extractions[arm].append(extraction)
        punctual_means[arm] = np.mean(punctual_extractions[arm])

        if i % 100 == 0:
            print("step", i,
                  "global mean", np.mean(extractions),
                  "arm performance", punctual_means)


if __name__ == "__main__":
    from GenerateSamples import GenerateBinarySamples, BinaryGroupParams

    real_group_specs = [BinaryGroupParams(0.01, 100000),
                        BinaryGroupParams(0.02, 100000),
                        ]

    data = [GenerateBinarySamples(real_group_specs).get_samples_for_group(i) for i in
            range(len(real_group_specs))]

    tests = BayesianABTestRunner("ucb1")
    tests.run_test(epsilon=0.05, data_samples=data, n_steps=10001)

    tests = BayesianABTestRunner("bandits")
    tests.run_test(epsilon=0.01, data_samples=data, n_steps=10001)
