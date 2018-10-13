import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ResNet import ResNet
import argparse
from utils import *
import time
from common.utils import allocate_gpu


def find_next_time(path_list, default=-1):
    if default > -1:
        return default
    run_times = [int(path.split('_')[0]) for path in path_list]
    # last_time = max(run_times)
    if default == -1:
        next_time = max(run_times) + 1 if run_times else 0
        return next_time
    elif default == -2:
        return max(run_times) if run_times else 0


def find_next_time(path_list,default=-1):
    run_times=[int(path.split('_')[0]) for path in path_list ]
    # print(run_times)
    last_time=max(run_times) if run_times else 0
    if default == -1:
        return last_time+1
    else:
        return default


"""parsing and configuration"""
def parse_args():

    GPU = -1
    GPU_ID = allocate_gpu(GPU)
    print('Using GPU %d'%GPU_ID)
    gpuNo = 'gpu10_%d'%GPU_ID
    
    optimizer_name='adashift' #adam adashift amsgrad sgd
    lr=0.01
    beta1=0.9
    beta2=0.999
    keep_num=10
    pred_g_op='max'
    epoch_num = 50

    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, mnist, fashion-mnist, tiny')
    parser.add_argument('--epoch', type=int, default=epoch_num, help='The number of epochs to run')
    parser.add_argument('--test_span', type=int, default=20, help='step interval for test')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')
    parser.add_argument('--gpuNo', type=str, default=gpuNo, help='which gpu to use')
    parser.add_argument('--run_time', type=int, default=-1, help="which time to run this experiment, used in the identifier of experiment. -1 automaticly add one to last time, -2 keep last record")
    # parser.add_argument('--GPU', type=int, default=-1, help="which gpu to use")

    # parser.add_argument('--T', type=str, default=T, help='identifier of experiment')
    parser.add_argument('--optimizer_name', type=str, default=optimizer_name, help='[sgd, adam, amsgrad, adashift')
    parser.add_argument('--lr', type=float, default=lr, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=beta1, help='beta1 for optimizer')
    parser.add_argument('--beta2', type=float, default=beta2, help='beta2 for optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for optimizer')
    parser.add_argument('--keep_num', type=int, default=keep_num, help='keep_num for adashift optimizer')
    parser.add_argument('--pred_g_op', type=str, default=pred_g_op, help='pred_g_op for adashift optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default="",
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # # --checkpoint_dir
    # check_folder(args.checkpoint_dir)
    #
    # # --result_dir
    # check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    # return args
    if args is None:
      exit()
    run_time=find_next_time(os.listdir('./logs'),args.run_time)
    T='%d_%s_%s_%d_%.3f_%.2f_%.3f'%(run_time,args.optimizer_name,args.pred_g_op,args.keep_num,args.lr,args.beta1,args.beta2)
    args.T = T
    print('Check params: %s'%T) 

    if args.run_time ==-1:
        time.sleep(6)
    log_dir='./logs/%s'%T
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir='./checkpoints/model_%s'%T
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.log_dir = log_dir
    args.checkpoint_dir = checkpoint_dir

    # open session
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        cnn = ResNet(sess, args)

        # build graph
        cnn.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            result=cnn.train()
            print(" [:)] Training finished! \n")

            cnn.test()
            print(" [:)] Test finished!")

        if args.phase == 'test' :
            cnn.test()
            print(" [:)] Test finished!")