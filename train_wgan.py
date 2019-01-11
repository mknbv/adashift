"""
The code borrowed from https://github.com/anibali/wgan-cifar10
"""
import os
import argparse

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import torchvision.utils
from torchvision.transforms import transforms

from adashift.optimizers import AdaShift
from wgan.inception_score import inception_score
from wgan.logger import Logger
from wgan.model import Generator, Discriminator
from wgan import lipschitz, progress


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_disc_gradients(discriminator, generator, real_var,
                             lipschitz_constraint,
                             spherical_noise=False):
    '''Calculate gradients and loss for the discriminator.'''

    # Enable gradient calculations for discriminator parameters
    for param in discriminator.parameters():
        param.requires_grad = True

    # Set discriminator parameter gradients to zero
    discriminator.zero_grad()

    lipschitz_constraint.prepare_discriminator()

    real_out = discriminator(real_var).mean()
    real_out.backward(torch.tensor(-1., device=get_device()))

    # Sample Gaussian noise input for the generator
    if spherical_noise:
      noise = np.random.randn(real_var.size(0), 100)
      noise /= np.linalg.norm(noise, axis=1, keepdims=True)
      noise = Variable(torch.tensor(noise.astype(np.float32),
                                    device=get_device()))
    else:
      noise = torch.randn(real_var.size(0), 128).type_as(real_var.data)
      noise = Variable(noise, volatile=True)

    gen_out = generator(noise)
    fake_var = Variable(gen_out.data)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.tensor(1., device=get_device()))

    loss_penalty = lipschitz_constraint.calculate_loss_penalty(real_var.data, fake_var.data)

    disc_loss = fake_out - real_out + loss_penalty


    return disc_loss


def calculate_gen_gradients(discriminator, generator, batch_size,
                           spherical_noise=False):
    '''Calculate gradients and loss for the generator.'''

    # Disable gradient calculations for discriminator parameters
    for param in discriminator.parameters():
        param.requires_grad = False

    # Set generator parameter gradients to zero
    generator.zero_grad()

    # # Sample Gaussian noise input for the generator
    if spherical_noise:
      noise = np.random.randn(batch_size, 100)
      noise /= np.linalg.norm(noise, axis=1, keepdims=True)
      noise = Variable(torch.tensor(noise.astype(np.float32),
                                    device=get_device()))
    else:
      noise = torch.randn(batch_size, 128).cuda()
      noise = Variable(noise)

    fake_var = generator(noise)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.tensor(-1., device=get_device()))

    gen_loss = -fake_out
    return gen_loss


def loop_data_loader(data_loader):
    '''Create an infinitely looping generator for a data loader.'''

    while True:
        for batch,l in data_loader:
            yield batch, l


def compute_inception_score(generator, nimages=int(30e3),
                            generator_batch_size=128, inception_batch_size=8):
    images = []
    cpu = torch.device("cpu")
    with torch.no_grad():
      for i in range(0, nimages, generator_batch_size):
        progress.bar(i, nimages, 'Generating images for inception score')
        nsamples = (generator_batch_size
                    - max(i + generator_batch_size - nimages, 0))
        noise = torch.randn(nsamples, 128).cuda()
        noise = Variable(noise)
        newimages = generator(noise)
        images.append(generator(noise).to(cpu))
      images = torch.cat(images)
    return inception_score(images, batch_size=inception_batch_size,
                           resize=True, splits=10)


def parse_args():
    '''Parse command-line arguments.'''

    parser = argparse.ArgumentParser(description='WGAN model trainer.')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
        help='number of epochs to train (default=1000)')
    parser.add_argument('--gen-iters', type=int, default=100, metavar='N',
        help='generator iterations per epoch (default=100)')
    parser.add_argument('--disc-iters', type=int, default=5, metavar='N',
        help='discriminator iterations per generator iteration (default=5)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size (default=128)')
    parser.add_argument('--disc-lr', type=float, default=2e-4, metavar='LR',
        help='discriminator learning rate (default=2e-4)')
    parser.add_argument('--gen-lr', type=float, default=2e-4, metavar='LR',
        help='generator learning rate (default=2e-4)')
    parser.add_argument('--unimproved', default=False, action='store_true',
        help='disable gradient penalty and use weight clipping instead')
    parser.add_argument('--generator-optimizer',
                        choices=[None, "adam", "adashift", "amsgrad"],
                        help="optimizer for generator")
    parser.add_argument('--discriminator-optimizer',
                        choices=["adam", "adashift", "amsgrad"],
                        help='optimizer for discriminator')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='log (default=64)')
    parser.add_argument('--spherical-noise', action="store_true")
    parser.add_argument('--penalty-coef', type=float, default=10.)

    args = parser.parse_args()

    return args


def get_optimizer(name):
  optimizers = {
      "adashift": AdaShift,
      "amsgrad": lambda *args, **kwargs: optim.Adam(*args, amsgrad=True,
                                                    **kwargs),
      "adam": optim.Adam
  }
  return optimizers[name]


def main():
    '''Main entrypoint function for training.'''

    # Parse command-line arguments
    args = parse_args()
    fmt = {'disc_loss':'.5e', 'gen_loss':'.5e' }
    logger_name = (
        "wgan-train_"
        f"{args.generator_optimizer}-{args.discriminator_optimizer}")
    logger = Logger(logger_name, fmt=fmt)
    logger_disc = Logger(logger_name+"_discriminator", fmt=fmt)
    logger_gen = Logger(logger_name+"_generator", fmt=fmt)

    # Create directory for saving outputs
    os.makedirs('out', exist_ok=True)

    # Initialise CIFAR-10 data loader
    # train_loader = DataLoader(torchvision.datasets.CIFAR10('./data/cifar-10'),
    #     args.batch_size, num_workers = 4, pin_memory = True, drop_last = True)


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    inf_train_data = loop_data_loader(train_loader)

    # Build neural network models and copy them onto the GPU
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    # Select which Lipschitz constraint to use
    if args.unimproved:
        lipschitz_constraint = lipschitz.WeightClipping(discriminator)
    else:
        lipschitz_constraint = lipschitz.GradientPenalty(discriminator,
                                                         args.penalty_coef)

    # Initialise the parameter optimisers
    optim_gen = (
        get_optimizer(args.generator_optimizer)(generator.parameters(),
                                                lr=2e-4, betas=(0., 0.999))
        if args.generator_optimizer else None)
    optim_disc = get_optimizer(args.discriminator_optimizer)(
        discriminator.parameters(), lr=2e-4, betas=(0., 0.999))

    i,j = 0, 0
    # Run the main training loop
    for epoch in range(args.epochs):
        avg_disc_loss = 0
        avg_gen_loss = 0

        for gen_iter in range(args.gen_iters):
            # Train the discriminator (aka critic)
            for _ in range(args.disc_iters):
                inputs, labels = next(inf_train_data)
                inputs.requires_grad = True
                real_var = inputs.cuda()

                disc_loss = calculate_disc_gradients(
                    discriminator, generator, real_var, lipschitz_constraint,
                    spherical_noise=args.spherical_noise)
                avg_disc_loss += disc_loss.item()
                optim_disc.step()
                if i % args.log_interval == 0:
                    logger_disc.add_scalar(i, 'disc_loss', disc_loss.item())
                i += 1

            # Train the generator
            if optim_gen:
              gen_loss = calculate_gen_gradients(
                  discriminator, generator, args.batch_size,
                  spherical_noise=args.spherical_noise)
              avg_gen_loss += gen_loss.item()
              optim_gen.step()
              if j % args.log_interval == 0:
                  logger_gen.add_scalar(j, 'gen_loss', gen_loss.item())
                  pass
              j += 1

            # # Save generated images
            # torchvision.utils.save_image((generator.last_output.data.cpu() + 1) / 2,
            #     'out/samples.png', nrow=8, range=(-1, 1))

            # Advance the progress bar
            progress.bar(gen_iter + 1, args.gen_iters,
                prefix='Epoch {:4d}'.format(epoch), length=30)
        # Calculate mean losses
        avg_disc_loss /= args.gen_iters * args.disc_iters
        avg_gen_loss /= args.gen_iters
        logger.add_scalar(epoch, 'gen_loss', avg_gen_loss)
        logger.add_scalar(epoch, 'disc_loss', avg_disc_loss)

        if optim_gen:
          inception_score = compute_inception_score(
              generator, generator_batch_size=args.batch_size)
          logger.add_scalar(epoch, "inception_score_mean", inception_score[0])
          logger.add_scalar(epoch, "inception_score_std", inception_score[1])

        logger_disc.save()
        if optim_gen:
          logger_gen.save()
        logger.save()
        # Print loss metrics for the last batch of the epoch
        printlog = (f"\nepoch {epoch}:"
                    f" disc_loss={disc_loss:8.4f}")
        if optim_gen:
          printlog += (f" gen_loss={gen_loss:8.4f}"
                       f" inception_score={inception_score[0]:8.4f}")
        print(printlog)

        # Save the discriminator weights and optimiser state
        torch.save({
            'epoch': epoch + 1,
            'model_state': discriminator.state_dict(),
            'optim_state': optim_disc.state_dict(),
        }, os.path.join('out',
                        args.discriminator_optimizer +  '_discriminator.pth'))

        # Save the generator weights and optimiser state
        savedict = {
            'epoch': epoch + 1,
            'model_state': generator.state_dict(),
        }
        if optim_gen:
          savedict["optim_state"] = optim_gen
        torch.save(savedict, os.path.join(
            'out', args.generator_optimizer+'_generator.pth'))

if __name__ == '__main__':
    main()
