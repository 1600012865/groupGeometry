import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default='sagan_1')
    parser.add_argument('--model',type=str, default='sagan')

    # Training setting
    parser.add_argument('--total_step', type=int, default=20000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--g_batch_size', type=int, default=2048)
    parser.add_argument('--d_batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--step_g', type=int, default=1)
    parser.add_argument('--step_d', type=int, default=1)     

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='cifar', choices=['lsun', 'celeb'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data/rec_three')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='/data-174/xuanjc/geometry/models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')
    parser.add_argument('--losscurve_path',type=str,default='./loss_curve')

    # Step size
    parser.add_argument('--log_step', type=int, default=40)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=5.0)
    parser.add_argument('--noise_step', type=int, default=100)
    parser.add_argument('--grad_step', type=int,default=100)

    #noise setting
    parser.add_argument('--max_p',type=float,default=0.5)
    parser.add_argument('--min_p',type=float,default=0)
    parser.add_argument('--max_std',type=float,default=40)
    parser.add_argument('--min_std',type=float,default=0)

    return parser.parse_args()