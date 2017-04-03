import numpy as np


class BayesianABTestRunner(object):
    """Epsilon greedy approach to run tests"""

    def __init__(self, epsilon):
        """
        Args:
                epsilon (Double): the epsilon to use
        """

        assert (0.0 <= epsilon <= 1.0)
        self.epsilon = epsilon


    def run_test(self, data_samples, n_steps):

        assert n_steps > 0
        for group in data:
            assert issubclass(group.dtype.type, np.integer)
            assert set(np.unique(group)) == set((0, 1))

        n_groups = len(data_samples)
        punctual_means = np.zeros((n_groups, ))
        punctual_extractions = [list([]) for _ in range(n_groups)]

        idxs = np.zeros((n_groups, ), dtype=int)

        extractions = []

        for i in range(n_steps):
            r = np.random.rand()

            if r < self.epsilon:
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


if __name__ == "__main__":
    from GenerateSamples import GenerateBinarySamples, BinaryGroupParams

    real_group_specs = [BinaryGroupParams(0.01, 100000),
                        BinaryGroupParams(0.02, 100000),
                        ]

    data = [GenerateBinarySamples(real_group_specs).get_samples_for_group(i) for i in
            range(len(real_group_specs))]

    tests = EpsGreedyRunner(epsilon=0.05)
    tests.run_test(data, 10001)



