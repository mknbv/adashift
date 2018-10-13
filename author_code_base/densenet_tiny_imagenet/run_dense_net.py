import argparse

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}

train_params_tiny = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'validation_set': True,
    'validation_split': None,
    'shuffle': 'every_epoch',
    'normalization': 'by_chanels',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn
    if name == 'Tiny':
        return train_params_tiny


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int,
        default=40,
        help='Depth of whole network')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN', 'Tiny'],
        default='C10',
        help='What dataset should be used')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    
    parser.add_argument(
        '--num_inter_threads', '-inter', type=int, default=1, metavar='',
        help='number of inter threads for inference / test')
    parser.add_argument(
        '--num_intra_threads', '-intra', type=int, default=128, metavar='',
        help='number of intra threads for inference / test')

    ## myself parameters
    parser.add_argument(
        '--exp_name', type=str, default="",
        help='name(identifier) of experiment')
    parser.add_argument(
        '--optimizer', type=str, default="adam",
        help='adam | amsgrad | adaShift')
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='initial learning rate')
    parser.add_argument(
        '--beta1', type=float, default=0.9,
        help='beta1 of adam | adaShift | amsgrad optimizer')
    parser.add_argument(
        '--beta2', type=float, default=0.999,
        help='beta2 of adam | adashift | amsgrad optimizer"')
    parser.add_argument(
        '--epsilon', type=float, default=1e-8,
        help='epsilon of adam | adashift | amsgrad optimizer')
    parser.add_argument(
        '--keep_num', type=int, default=10,
        help='keep_num of adashift optimizer')
    parser.add_argument(
        '--use_mov', type=int, default=0,
        help='set to 0, not use move in adashift optimizer; '
             'set to 1, use move in adashift optimizer')
    parser.add_argument(
        '--mov_num', type=int, default=10,
        help='mov_num of adashift optimizer')
    parser.add_argument(
        '--pred_g_op', type=str, default="max",
        help='pred_g_op of adashift optimizer')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='batch size')
    parser.add_argument(
        '--n_epochs', type=int, default=300,
        help='# of training epoch')
    parser.add_argument(
        '--reduce_lr_epoch_1', type=int, default=150,
        help='which epoch to first reduce learning rate')
    parser.add_argument(
        '--reduce_lr_epoch_2', type=int, default=225,
        help='which epoch to second reduce learning rate')
    parser.add_argument('--normalization', type=str, choices=[None, "divide_256", "divide_255", "by_chanels"],
                        default="by_chanels", help='type of normalization applied to input')

    parser.set_defaults(renew_logs=True)

    args = parser.parse_args()

    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN', 'Tiny']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True
    if args.optimizer in ["adam", "amsgrad", "adaShift"]:
        args.weight_decay = 0

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    train_params["initial_learning_rate"] = args.lr
    train_params["batch_size"] = args.batch_size
    train_params["n_epochs"] = args.n_epochs
    train_params["reduce_lr_epoch_1"] = args.reduce_lr_epoch_1
    train_params["reduce_lr_epoch_2"] = args.reduce_lr_epoch_2
    train_params["normalization"] = args.normalization

    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model..")
    model = DenseNet(data_provider=data_provider, **model_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
