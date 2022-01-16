import math

def priors_specs():
    """
    Create the specifications for all the priors.

    """

    conv1_aratio = [1, 2, 1/2, 3/2]
    conv_1 = {"fdim": 38, "scale": 0.1, "aratios": conv1_aratio, "num": 4, "total": 5776}  # 38 x 38 x 4 = 5776

    conv2_aratio = [1, 2, 1/2, 3, 1/3, 2.5, 1/2.5]
    conv_2 = {"fdim": 19, "scale": 0.2, "aratios": conv2_aratio, "num": 7, "total": 2527}  # 19 x 19 x 7 = 2527

    conv3_aratio = [1, 2, 1/2, 3, 1/3, 1/2.5]
    conv_3 = {"fdim": 10, "scale": 0.38, "aratios": conv3_aratio, "num": 6, "total": 600} # 10 x 10 x 6 = 600 

    conv4_aratio = [1, 2, 1/2, 3, 1/3, 2.5, 1/2.5]
    conv_4 = {"fdim": 5, "scale": 0.55, "aratios": conv4_aratio, "num": 7, "total": 175}  # 5 x 5 x 7 = 175

    conv5_aratio = [1, 2, 1/2, 3, 1/3]
    conv_5 = {"fdim": 3, "scale": 0.725, "aratios": conv5_aratio, "num": 5, "total": 45} # 3 x 3 x 5 = 45

    conv6_aratio = [1, 2, 1/2, 3, 1/3]
    conv_6 = {"fdim": 1, "scale": 0.9, "aratios": conv6_aratio, "num": 5, "total": 5}   # 1 x 1 x 5 = 5

    total_prior_num = 9128 # Total: 9128

    return {'conv_1':conv_1,'conv_2': conv_2,'conv_3': conv_3,'conv_4': conv_4,'conv_5': conv_5,'conv_6': conv_6}, total_prior_num

# --- Helper functions ---------------------------------------------------

def prior_dim(scale, aspect_ratio):
    w = scale * math.sqrt(aspect_ratio)
    h = scale / math.sqrt(aspect_ratio)

    return w, h


def create_prior_aux(scale, aspect_ratios):
    priors = []

    for aratio in aspect_ratios:
        priors.append(prior_dim(scale, aratio))

    return priors

# ------------------------------------------------------------------------

def create_prior(prior_conv : str):
    specs = priors_specs()[0][prior_conv]

    scale = specs['scale']
    aspect_ratios = specs['aratios']

    priors = create_prior_aux(scale, aspect_ratios)

    return priors


def test():
    priors = create_prior('conv_1')
    print(priors)

test()
