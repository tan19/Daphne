import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

__all__ = ['BlackScholes']

class BlackScholes():
    def __init__(self):
        self.spot = np.array([100])
        self.strike = np.array([100])
        self.tenor = np.array([1 / 12]) # one month

        self.r = np.array([0.20])

        self.borrow_rate = np.array([0.00])
        self.div = np.array([0.00])
        
        self.vol = np.array([0.16])
    
    def plot(self, prop, values):
        current_values = self.__dict__[prop]
        self.__dict__[prop] = values

        plt.subplot(2, 2, 1) # row 1, col 2 index 1
        plt.plot(self.__dict__[prop], self.delta)
        plt.title("Delta")
        plt.xlabel(prop)
        plt.ylabel('Delta')

        plt.subplot(2, 2, 2) # row 1, col 2 index 1
        plt.plot(self.__dict__[prop], self.gamma)
        plt.title("Gamma")
        plt.xlabel(prop)
        plt.ylabel('Gamma')

        plt.subplot(2, 2, 3) # row 1, col 2 index 1
        plt.plot(self.__dict__[prop], self.theta)
        plt.title("Theta")
        plt.xlabel(prop)
        plt.ylabel('Theta')

        plt.subplot(2, 2, 4) # row 1, col 2 index 1
        plt.plot(self.__dict__[prop], self.vega)
        plt.title("Vega")
        plt.xlabel(prop)
        plt.ylabel('Vega')

        self.__dict__[prop] = current_values # restore

    @property
    def d1(self):
        return (np.log(self.forward / self.strike) + 0.5 * (self.vol ** 2) * self.tenor) / (self.vol * np.sqrt(self.tenor))

    @property
    def pdf_d1(self):
        return norm.pdf(self.d1)

    @property
    def cdf_d1(self):
        return norm.cdf(self.d1)

    @property
    def d2(self):
        return self.d1 - (self.vol ** 2) * self.tenor

    @property
    def pdf_d2(self):
        return norm.pdf(self.d2)

    @property
    def cdf_d2(self):
        return norm.cdf(self.d2)

    @property
    def discount_factor(self):
        return np.exp(-self.r * self.tenor)

    @property
    def forward(self):
        return self.spot * np.exp(self.r * self.tenor)

    @property
    def pv(self):
        return self.discount_factor * (self.forward * self.cdf_d1 - self.strike * self.cdf_d2)

    @property
    def delta(self):
        return (np.exp(-self.borrow_rate * self.tenor) * self.cdf_d1)

    @property
    def gamma(self):
        return np.exp(-self.borrow_rate * self.tenor) * self.pdf_d1 / (self.spot * self.vol * np.sqrt(self.tenor))

    @property
    def theta(self):
        part_1 = -np.exp(-self.borrow_rate * self.tenor) * self.spot * self.pdf_d1 * self.vol / (2 * np.sqrt(self.tenor))
        part_2 = -self.r * self.strike * np.exp(-self.r * self.tenor) * self.cdf_d2
        part_3 = self.borrow_rate * self.spot * np.exp(-self.borrow_rate * self.tenor) * self.cdf_d1
        return part_1 + part_2 + part_3

    @property
    def vega(self):
        return self.spot * np.exp(-self.borrow_rate * self.tenor) * self.pdf_d1 * np.exp(self.tenor)
