import math
import numpy as np
import torch

FEATURE_LAYERS = ['feature_1', 'feature_2',
                  'feature_3', 'feature_4', 'feature_5', 'feature_6']


def priors_specs():
    """
    Create the specifications for all the priors.

    """

    conv1_aratio = [1, 2, 1/2, 3/2]
    feature_1 = {"fdim": 38, "scale": 0.1, "aratios": conv1_aratio,
                 "num": 4, "total": 5776}  # 38 x 38 x 4 = 5776

    conv2_aratio = [1, 2, 1/2, 3, 1/3, 2.5, 1/2.5]
    feature_2 = {"fdim": 19, "scale": 0.2, "aratios": conv2_aratio,
                 "num": 7, "total": 2527}  # 19 x 19 x 7 = 2527

    conv3_aratio = [1, 2, 1/2, 3, 1/3, 1/2.5]
    feature_3 = {"fdim": 10, "scale": 0.38, "aratios": conv3_aratio,
                 "num": 6, "total": 600}  # 10 x 10 x 6 = 600

    conv4_aratio = [1, 2, 1/2, 3, 1/3, 2.5, 1/2.5]
    feature_4 = {"fdim": 5, "scale": 0.55, "aratios": conv4_aratio,
                 "num": 7, "total": 175}  # 5 x 5 x 7 = 175

    conv5_aratio = [1, 2, 1/2, 3, 1/3]
    feature_5 = {"fdim": 3, "scale": 0.725, "aratios": conv5_aratio,
                 "num": 5, "total": 45}  # 3 x 3 x 5 = 45

    conv6_aratio = [1, 2, 1/2, 3, 1/3]
    feature_6 = {"fdim": 1, "scale": 0.9, "aratios": conv6_aratio,
                 "num": 5, "total": 5}   # 1 x 1 x 5 = 5

    total_prior_num = 9128  # Total: 9128

    return {'feature_1': feature_1, 'feature_2': feature_2, 'feature_3': feature_3, 'feature_4': feature_4, 'feature_5': feature_5, 'feature_6': feature_6}, total_prior_num


def create_priors():

    specs = priors_specs()[0]
    all_priors = np.empty((0, 4))

    for layer in FEATURE_LAYERS:

        fdim = specs[layer]['fdim']
        aratios = specs[layer]['aratios']
        scale = specs[layer]['scale']

        indecies = np.indices((fdim, fdim)).reshape(2, -1)

        X = indecies[0, :]
        Y = indecies[1, :]

        CX = (X + 0.5) / fdim
        CY = (Y + 0.5) / fdim

        feature_prior = np.empty((1, 4))

        for aratio in aratios:
            W = np.full((CX.shape[0], 1), scale * math.sqrt(aratio))
            H = np.full((CX.shape[0], 1), scale / math.sqrt(aratio))

            feature_prior = np.column_stack((CX, CY, W, H))
            all_priors = np.vstack((all_priors, feature_prior))

    all_priors = torch.from_numpy(all_priors).clamp(0, 1)

    return all_priors


def test():
    all_priors = create_priors()
    print(all_priors[-1])
