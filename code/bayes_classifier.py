import numpy as np
import util
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture


class BayesClassifier:
    def fit(self, x, y):
        self.N = len(np.unique(y))
        self.models = []
        for n in range(self.N):
            x_n = x[y == n]
            mean = np.mean(x_n.T, axis=1)
            cov = np.cov(x_n.T)
            model = dict(mean=mean, cov=cov)
            self.models.append(model)

    def sample_given_y(self, y):
        model = self.models[y]
        x = np.random.multivariate_normal(model["mean"], model["cov"])
        return x

    def sample(self):
        y = np.random.randint(self.N)
        return self.sample_given_y(y)


class BayesMGGClassifier:
    def fit(self, x, y):
        self.N = len(np.unique(y))
        self.models = []
        for n in range(self.N):
            x_n = x[y == n]
            model = BayesianGaussianMixture(10)
            model.fit(x_n)
            self.models.append(model)

    def sample_given_y(self, y):
        model = self.models[y]

        sample, cluster = model.sample()

        return sample

    def sample(self):
        y = np.random.randint(self.N)
        return self.sample_given_y(y)


if __name__ == "__main__":
    Xtrain, Ytrain, Xtest, Ytest = util.getKaggleMNIST()
    N = len(np.unique(Ytrain))
    clf = BayesClassifier()
    clf = BayesMGGClassifier()
    clf.fit(Xtrain, Ytrain)

    for number in range(N):
        sample = clf.sample_given_y(number)
        # mean = clf.models[number]["mean"]

        plt.imshow(sample.reshape(28, 28), cmap="gray")
        plt.show()
        # plt.imshow(mean.reshape(28, 28), cmap="gray")
        # plt.show()

