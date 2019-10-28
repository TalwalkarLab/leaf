import numpy as np

from scipy.special import softmax


NUM_DIM = 10

class SyntheticDataset:

    def __init__(
            self,
            num_classes=2,
            seed=931231,
            num_dim=NUM_DIM,
            prob_clusters=[0.5, 0.5]):

        np.random.seed(seed)

        self.num_classes = num_classes
        self.num_dim = num_dim
        self.num_clusters = len(prob_clusters)
        self.prob_clusters = prob_clusters

        self.side_info_dim = self.num_clusters

        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes, self.side_info_dim))

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1)**(-1.2)

        self.means = self._generate_clusters()

    def get_task(self, num_samples):
        cluster_idx = np.random.choice(
            range(self.num_clusters), size=None, replace=True, p=self.prob_clusters)
        new_task = self._generate_task(self.means[cluster_idx], cluster_idx, num_samples)
        return new_task

    def _generate_clusters(self):
        means = []
        for i in range(self.num_clusters):
            loc = np.random.normal(loc=0, scale=1., size=None)
            mu = np.random.normal(loc=loc, scale=1., size=self.side_info_dim)
            means.append(mu)
        return means

    def _generate_x(self, num_samples):
        B = np.random.normal(loc=0.0, scale=1.0, size=None)
        loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)

        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, cluster_mean):
        model_info = np.random.normal(loc=cluster_mean, scale=0.1, size=cluster_mean.shape)
        w = np.matmul(self.Q, model_info)
        
        num_samples = x.shape[0]
        prob = softmax(np.matmul(x, w) + np.random.normal(loc=0., scale=0.1, size=(num_samples, self.num_classes)), axis=1)
                
        y = np.argmax(prob, axis=1)
        return y, w, model_info

    def _generate_task(self, cluster_mean, cluster_id, num_samples):
        x = self._generate_x(num_samples)
        y, w, model_info = self._generate_y(x, cluster_mean)

        # now that we have y, we can remove the bias coeff
        x = x[:, 1:]

        return {'x': x, 'y': y, 'w': w, 'model_info': model_info, 'cluster': cluster_id}