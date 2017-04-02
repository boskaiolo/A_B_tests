from collections import namedtuple
import numpy as np


RealGroupParams = namedtuple('RealGroupParams', ['mean', 'var', 'length'])
BinaryGroupParams = namedtuple('BinaryGroupParams', ['mean', 'length'])

class GenerateRealSamples(object):
    """Generates real samples for 1+ groups"""

    def __init__(self, group_specs):
        """Set up the parameters

        Args:
            group_specs (List[RealGroupParams]): Each namedtuple correspond to a group
        """
        assert isinstance(group_specs, list)
        self.group_specs = group_specs

    def get_samples_for_group(self, n):
        """Return samples as indicated in the specs

        Args:
            n (Int, Long): Group number

        Note:
            Each call is independent; results may vary between consecutive calls
        """
        if n >= len(self.group_specs):
            raise IndexError("I only have {} group specs".format(len(self.group_specs)))

        specs = self.group_specs[n]
        if specs.var <= 0:
            raise ValueError("Variance must be > 0")

        return np.sqrt(specs.var) * np.random.randn(specs.length) + specs.mean


class GenerateBinarySamples(object):
    """Generates binary samples for 1+ groups"""

    def __init__(self, group_specs):
        """Set up the parameters

        Args:
            group_specs (List[BinaryGroupParams]): Each namedtuple correspond to a group
        """
        assert isinstance(group_specs, list)
        self.group_specs = group_specs

    def get_samples_for_group(self, n):
        """Return samples as indicated in the specs

        Args:
            n (Int, Long): Group number

        Note:
            Each call is independent; results may vary between consecutive calls
        """
        if n >= len(self.group_specs):
            raise IndexError("I only have {} group specs".format(len(self.group_specs)))

        specs = self.group_specs[n]

        if specs.mean < 0 or specs.mean > 1:
            raise ValueError("Mean should be between 0 and 1")

        n_1 = int(specs.length*specs.mean)
        n_0 = int(specs.length-n_1)

        return np.random.permutation(np.concatenate((np.zeros(n_0, dtype=int), np.ones(n_1, dtype=int))))


if __name__ == "__main__":
    real_group_specs = [RealGroupParams(0, 1, 1000000),
                        RealGroupParams(100, 1, 1000000),
                        RealGroupParams(50, 50, 1000000)]
    real_samples = GenerateRealSamples(real_group_specs)

    for i, el in enumerate(real_group_specs):
        a_sample = real_samples.get_samples_for_group(i)

        assert np.isclose(np.mean(a_sample), el.mean, atol=0.1)
        assert np.isclose(np.var(a_sample), el.var, atol=0.1)
        assert len(a_sample) == el.length

    binary_group_specs = [BinaryGroupParams(0.1, 100000)]
    binary_samples = GenerateBinarySamples(binary_group_specs)

    for i, el in enumerate(binary_group_specs):
        a_sample = binary_samples.get_samples_for_group(i)
        assert np.isclose(np.mean(a_sample), el.mean, atol=1E-6)
        assert len(a_sample) == el.length

