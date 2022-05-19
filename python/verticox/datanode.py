import numpy as np
from flask import Flask
from flask_restful import Resource
from typing import Optional

RHO = 0.25


class DataNode(Resource):
    def __init__(self, covariates: Optional[np.array] = None, events: Optional[np.array] = None,
                 rho: float = RHO):
        self.covariates = covariates
        self.events = events

        # Local update
        self.rho = rho

        # Parts that stay constant over iterations
        # Square all covariates and sum them together
        # The formula says for every patient, x needs to be multiplied by itself.
        # Squaring all covariates with themselves comes down to the same thing since x_nk is supposed to
        # be one-dimensional
        self.covariates_multiplied= (covariates * covariates.transpose()).sum(axis=0)
        self.covariates_sum= covariates.sum(axis=0)

    def put(self, z, gamma):
        sigma = DataNode.local_update(self.covariates, z, gamma, self.rho,
                                      self.covariates_multiplied, self.covariates_sum)


    @staticmethod
    def sum_covariates(covariates: np.array):
        return np.sum(covariates, axis=0)

    @staticmethod
    def multiply_covariates(covariates: np.array):
        return np.square(covariates).sum()

    @staticmethod
    def elementwise_multiply_sum(one_dim: np.array, two_dim: np.array):
        """
        Every element in one_dim does elementwise multiplication with its corresponding row in two_dim.

        All rows of the result will be summed together vertically.
        """
        multiplied = np.zeros(two_dim.shape)
        for i in range(one_dim.shape[0]):
            multiplied[i] = one_dim[i] * two_dim[i]

        return multiplied.sum(axis=0)

    @staticmethod
    def compute_beta(covariates: np.array, z: np.array, gamma: np.array, rho,
                     multiplied_covariates, covariates_sum):
        first_component = 1 / (rho * multiplied_covariates)

        pz = rho * z

        second_component = DataNode.elementwise_multiply_sum(pz - gamma, covariates) + \
                           covariates_sum

        return second_component / first_component

    @staticmethod
    def compute_sigma(beta, covariates):
        return np.matmul(covariates, beta)

    @staticmethod
    def local_update(covariates: np.array, z: np.array, gamma: np.array, rho,
                     covariates_multiplied, covariates_sum):
        beta = DataNode.compute_beta(covariates, z, gamma, rho, covariates_multiplied,
                                     covariates_sum)

        return DataNode.compute_sigma(beta, covariates)
