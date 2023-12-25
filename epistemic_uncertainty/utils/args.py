import argparse
from argparse import ArgumentParser

ARGS_GROUP = ['main', 'dataset', 'model', 'evaluation']


def get_parser() -> ArgumentParser:
    """
    Creates a parser for the arguments
    :return ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    # Main Arguments
    main_parser = parser.add_argument_group('main')
    main_parser.add_argument('--test', type=bool, default=False,
                             help='If true, loads a pretrained model and test it on test set.')
    main_parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to be used')
    # Dataset Arguments
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument('--dataset', type=str, default='Human36m', choices=['Human36m', 'AMASS', '3DPW'],
                                help='Specifies the dataset to be used.')
    dataset_parser.add_argument('--batch_size', type=int, default=64, help='Batch size of the dataloader')
    dataset_parser.add_argument('--num_workers', type=int, default=8, help='Number of workers of the dataloader')
    dataset_parser.add_argument('--dataset_path', type=str, default='human3_6', help='Path to the dataset')
    dataset_parser.add_argument('--fake_labeling', type=bool, default=False,
                                help='Determines if fake samples are needed to be generated. '
                                     'Only applicable to main and divided mode')
    dataset_parser.add_argument('--input_n', type=int, default=10, choices=[10, 50], help='Input sequence\'s length')
    dataset_parser.add_argument('--output_n', type=int, default=25, help='Output sequence\'s length')
    # Model Arguments
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument('--alpha', type=float, default=0.001, help='Alpha value for weighting L2 regularization')
    model_parser.add_argument('--lstm_optimizer', type=str, default='adam', help='LSTM model\'s optimizer')
    model_parser.add_argument('--lstm_scheduler', type=str, default='tri', help='LSTM model\'s scheduler')
    model_parser.add_argument('--lstm_lr', type=float, default=0.0001, help='LSTM model\'s learning rate')
    model_parser.add_argument('--lstm_lr_decay', type=float, default=0.99, help='Optimizer Learning Rate Decay '
                                                                                'Parameter')
    model_parser.add_argument('--lstm_epochs', type=int, default=200, help='Number of epochs for LSTM train')
    model_parser.add_argument('--lstm_path', type=str, default=None, help='Path to a trained LSTM model')
    model_parser.add_argument('--hidden_dim', type=int, default=512, help='Latent space dimension of the LSTM model')
    model_parser.add_argument('--encoded_dim', type=int, default=32, help='Latent space dimension of the '
                                                                          'FinalAutoEncoder model')
    model_parser.add_argument('--n_clusters', type=int, default=17, help='Number of clusters for DC model. You need to '
                                                                         'determine this with the specific experiment')
    model_parser.add_argument('--fake_clusters', type=int, nargs='+', default=None, help='List of indices of the fake '
                                                                                         'clusters')
    model_parser.add_argument('--k_init_batch', type=int, default=4, help='Batch size for the initial K-means for '
                                                                          'DC model training')
    model_parser.add_argument('--dc_lr', type=float, default=0.0005, help='DC model\'s learning rate')
    model_parser.add_argument('--dc_lr_decay', type=float, default=0.98, help='Learning rate decay factor per epoch')
    model_parser.add_argument('--dc_weight_decay', type=float, default=0.00001, help='DCEC optimizer Weight Decay '
                                                                                     'Parameter')
    model_parser.add_argument('--dc_gamma', type=float, default=0.6, help='Gamma values for weighting clustering loss')
    model_parser.add_argument('--dc_epochs', type=int, default=10, help='Number of epochs for DC train')
    model_parser.add_argument('--dc_stop_cret', type=int, default=0.001, help='Stop criteria value for DC model\'s '
                                                                              'train')
    model_parser.add_argument('--dc_update_interval', type=float, default=2.0,
                              help='Update interval for target distribution P. Float, for fractional update')
    model_parser.add_argument('--dc_path', type=str, default=None, help='Path to a trained DC model')
    model_parser.add_argument('--ae_path', type=str, default=None, help='Path to a trained FinalAutoEncoder model')
    model_parser.add_argument('--ae_epochs', type=int, default=30, help='Number of epochs for FinalAutoEncoder train')
    # Evaluation Arguments
    evaluation_parser = parser.add_argument_group('evaluation')
    evaluation_parser.add_argument('--model_dict_path', type=str, default=None,
                                   help='Path to a dictionary of outputs and ground-truths of a prediction model')
    evaluation_parser.add_argument('--model_path', type=str, default=None, help='Path to a prediction model')
    evaluation_parser.add_argument('--dc_model_path', type=str, default=None, help='Path to a pretrained dc_model.')
    evaluation_parser.add_argument('--output_path', type=str, default='output', help='Path to the output results')
    # TODO: Complete list of choices (Yashar, )
    evaluation_parser.add_argument('--pred_model', type=str, default=None, choices=['sts', ],
                                   help='Name of the prediction model to be evaluated')
    return parser


def get_args(parser: ArgumentParser) -> list:
    """
    Returns main, dataset, model and evaluation args from a given parser.
    :param parser:
    :return list:
    """
    args = parser.parse_args()
    separated_args = []

    for group in parser._action_groups:
        if group.title not in ARGS_GROUP:
            continue
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        separated_args.append(argparse.Namespace(**group_dict))
    return separated_args
