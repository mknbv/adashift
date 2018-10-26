'''
Classes for imposing a Lipschitz constraint on the discriminator of a WGAN.
'''

import torch
from torch import autograd

class LipschitzConstraint:
    '''Abstract base class for Lipschitz constraint implementations.'''

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def prepare_discriminator(self):
        '''Prepare the discriminator for an iteration.'''

        raise NotImplementedError()

    def calculate_loss_penalty(self, real_var, fake_var):
        '''Calculate a penalty for the discriminator loss.'''

        raise NotImplementedError()

class GradientPenalty(LipschitzConstraint):
    '''Gradient penalty Lipschitz constraint.

    Implements the penalty described in "Improved Training of Wasserstein GANs".
    '''

    def __init__(self, discriminator, coefficient=10):
        super().__init__(discriminator)

        self.coefficient = coefficient

    def prepare_discriminator(self):
        pass

    def calculate_loss_penalty(self, real_var, fake_var):
        assert real_var.size(0) == fake_var.size(0), \
            'expected real and fake data to have the same batch size'

        batch_size = real_var.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_var)
        alpha = alpha.type_as(real_var)

        interp_data = alpha * real_var + ((1 - alpha) * fake_var)
        interp_data = autograd.Variable(interp_data, requires_grad=True)

        disc_out = self.discriminator(interp_data)
        grad_outputs = torch.ones(disc_out.size()).type_as(disc_out.data)

        gradients = autograd.grad(
            outputs=disc_out,
            inputs=interp_data,
            grad_outputs=grad_outputs,
            create_graph=True,
            only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)

        gradient_penalty = self.coefficient * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()

        return gradient_penalty

class WeightClipping(LipschitzConstraint):
    '''Weight clipping Lipshitz constraint.

    Implements the approach originally described in "Wasserstein GAN".
    '''

    def __init__(self, discriminator, clamp_lower=-0.01, clamp_upper=0.01):
        super().__init__(discriminator)

        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper

    def prepare_discriminator(self):
        for param in self.discriminator.parameters():
            param.data.clamp_(self.clamp_lower, self.clamp_upper)

    def calculate_loss_penalty(self, real_var, fake_var):
        return 0
