"""
https://en.wikipedia.org/wiki/CMA-ES
"""
import numpy as np

class CmaES:
    def __init__(self, samples=2) -> None:
        self.samples = samples
        self.mean = np.ones(1)
        self.sigma = np.ones(1) * -1
        self.covariance = np.eye(1)

    def fit(self, fitness):
        for _ in range(1_000):
            results = np.zeros((self.samples))
            value = np.zeros((self.samples))
            num_parents = 1
            for i in range(self.samples):
                x_i = np.random.multivariate_normal(
                    mean=self.mean,
                    cov=self.covariance * (self.sigma ** 2)
                )
            #  print(x_i)
                x_i_fitness = fitness(x_i)
            #  print(x_i_fitness)
                results[i] = x_i_fitness
                value[i] = x_i
            index = results.argsort()[:num_parents]
         #   print(value[index].mean(keepdims=True), self.mean)
            self.mean = value[index].mean(keepdims=True)
        """
        https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
        
        """
        print(np.random.multivariate_normal(
            mean=self.mean,
            cov=self.covariance * (self.sigma ** 2)
        ))    
    
if __name__ == "__main__":
    def fitness(i):
        distance_from_desired = np.abs(i - 1)
        return distance_from_desired
    error = CmaES().fit(fitness)
    print(error)
