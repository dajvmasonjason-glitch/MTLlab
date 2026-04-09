import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods.weight_methods_add import METHODS


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--n-epochs", type=int, default=300)
common_parser.add_argument("--batch-size", type=int, default=120, help="batch size")
common_parser.add_argument(
    "--method", type=str, choices=list(METHODS.keys()), help="MTL weight method"
)
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument(
    "--method-params-lr",
    type=float,
    default=0.025,
    help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")
# NashMTL
common_parser.add_argument(
    "--nashmtl-optim-niter", type=int, default=20, help="number of CCCP iterations"
)
common_parser.add_argument(
    "--update-weights-every",
    type=int, default=1,
    help="update task weights every x iterations.",
)
# stl
common_parser.add_argument(
    "--main-task",
    type=int, default=0,
    help="main task for stl. Ignored if method != stl",
)
# cagrad
common_parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
# fairgrad
common_parser.add_argument("--alpha", type=float, default=1.0, help="alpha for FairGrad alg.")
# famo
common_parser.add_argument("--gamma", type=float, default=0.01, help="gamma of famo/ofamo/pfamo")
common_parser.add_argument("--gamma2", type=float, default=0.1, help="gamma2 of ofamo")
common_parser.add_argument("--gamma3", type=float, default=0.1, help="gamma3 of pfamo")
common_parser.add_argument("--cov_history_size", type=int, default=10, help="covariance history size of pfamo")
common_parser.add_argument("--use_log", action='store_true', help="whether use log for famo")
common_parser.add_argument("--max_norm", type=float, default=1.0, help="beta for RMS_weight alg.")
common_parser.add_argument("--task", type=int, default=0, help="train single task number for (celeba)")
# dwa
common_parser.add_argument("--dwa-temp",
    type=float, default=2.0,
    help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
)
# ldcmtl
common_parser.add_argument("--ldcmtl_lambda", type=float, default=0.1, help="lambda parameter for LDCMTL (论文推荐: CelebA 0.01, Cityscapes 0.1)")
common_parser.add_argument("--ldcmtl_alpha", type=float, default=0.1, help="alpha parameter for LDCMTL")
common_parser.add_argument("--ldcmtl_feat_dim", type=int, default=512, help="Feature dimension for router network in LDC-MTL")
common_parser.add_argument("--ldcmtl_tau_mode", type=str, default="ones", choices=["ones", "sigma"], help="Tau mode for loss discrepancy: 'ones' or 'sigma'")
# go4align
common_parser.add_argument("--num_groups", type=int, default=2, help="Number of groups for GO4Align clustering")
common_parser.add_argument("--go4align_robust_step_size", type=float, default=0.0001, help="Robust step size for GO4Align algorithm")
common_parser.add_argument("--go4align_gamma3", type=float, default=0.1, help="gamma3 for GO4AlignCov covariance penalty")
common_parser.add_argument("--go4align_history_size", type=int, default=15, help="History size for GO4AlignCov covariance calculation")
# ldcnew2
common_parser.add_argument("--ldcnew2_num_groups", type=int, default=2, help="Number of groups for LDCNew2 clustering")
common_parser.add_argument("--ldcnew2_robust_step_size", type=float, default=0.0001, help="Robust step size for LDCNew2 GO4Align algorithm")
# ldcnew3
common_parser.add_argument("--ldcnew3_eps", type=float, default=1e-8, help="Small constant epsilon for LDCNew3 smooth objective function")
common_parser.add_argument("--ldcnew3_lambda", type=float, default=0.1, help="Lambda parameter for LDCNew3 to control the weight of upper-level smooth objective (论文推荐: CelebA 0.01, Cityscapes 0.1)")
# ldcnew4
# famo_lbfgs
common_parser.add_argument("--famo_lbfgs_m", type=int, default=10, help="L-BFGS history buffer size for FamoLBFGS")
# pivrg
common_parser.add_argument("--pivrg_bound", type=float, default=2.0, help="Bound parameter for temperature calculation in PIVRG")
common_parser.add_argument("--pivrg_mintemp", type=float, default=10, help="Minimum temperature for softmax in PIVRG")
# consmtl
common_parser.add_argument("--consmtl_lambda", type=float, default=1.0, help="Regularization parameter for consensus loss in ConsMTL")
# rbldc
common_parser.add_argument("--rbldc_lambda", type=float, default=0.1, help="Lambda parameter for RB-LDC to control the weight of upper-level loss (论文推荐: CelebA 0.01, Cityscapes 0.1)")
common_parser.add_argument("--rbldc_alpha", type=float, default=0.1, help="Alpha parameter for RB-LDC")
common_parser.add_argument("--rbldc_feat_dim", type=int, default=512, help="Feature dimension for router network in RB-LDC")
common_parser.add_argument("--rbldc_gamma", type=float, default=0.01, help="Gamma parameter for RB-LDC loss rate calculation")
# grape
common_parser.add_argument("--grape_K", type=int, default=2, help="Number of groups for GRAPE clustering (default: 2)")
common_parser.add_argument("--grape_alpha", type=float, default=0.1, help="EMA smoothing coefficient for GRAPE indicator (default: 0.1)")
common_parser.add_argument("--grape_tau", type=float, default=1.0, help="Softmax temperature for GRAPE group weights (default: 1.0)")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
                max_norm=args.max_norm,
            ),
            stl=dict(main_task=args.main_task),
            dwa=dict(temp=args.dwa_temp),
            cagrad=dict(c=args.c, max_norm=args.max_norm),
            log_cagrad=dict(c=args.c, max_norm=args.max_norm),
            famo=dict(gamma=args.gamma,
                      w_lr=args.method_params_lr,
                      max_norm=args.max_norm),
            ofamo=dict(gamma=args.gamma,
                       gamma2=args.gamma2,
                       w_lr=args.method_params_lr,
                       max_norm=args.max_norm),
            pfamo=dict(gamma=args.gamma,
                       gamma3=args.gamma3,
                       cov_history_size=args.cov_history_size,
                       w_lr=args.method_params_lr,
                       max_norm=args.max_norm),
            corrfamo=dict(gamma=args.gamma,
                          w_lr=args.method_params_lr,
                          lambda_=getattr(args, 'consmtl_lambda', 0.1),
                          corr_history_size=getattr(args, 'cov_history_size', 10),
                          max_norm=args.max_norm),
            opfamo=dict(gamma=args.gamma,
                        gamma2=args.gamma2,
                        gamma3=args.gamma3,
                        cov_history_size=args.cov_history_size,
                        w_lr=args.method_params_lr,
                        max_norm=args.max_norm),
            fairgrad=dict(alpha=args.alpha, max_norm=args.max_norm),
            fast_fairgrad=dict(alpha=args.alpha, max_norm=args.max_norm),
            ldcmtl=dict(lambda_param=args.ldcmtl_lambda,
                       max_norm=args.max_norm,
                       feat_dim=args.ldcmtl_feat_dim,
                       tau_mode=args.ldcmtl_tau_mode),
            ldcnew1=dict(lambda_param=args.ldcmtl_lambda,
                        max_norm=args.max_norm,
                        feat_dim=args.ldcmtl_feat_dim,
                        tau_mode=args.ldcmtl_tau_mode),
            ldcnew2=dict(num_groups=getattr(args, 'ldcnew2_num_groups', 2),
                        robust_step_size=getattr(args, 'ldcnew2_robust_step_size', 0.0001),
                        max_norm=args.max_norm),
            ldcnew3=dict(max_norm=args.max_norm,
                        eps=getattr(args, 'ldcnew3_eps', 1e-8),
                        lambda_param=getattr(args, 'ldcnew3_lambda', 0.1)),
            ldcnew4=dict(max_norm=args.max_norm),
            cons_city=dict(max_norm=args.max_norm,
                        lambda_=args.consmtl_lambda),
            pivrg=dict(max_norm=args.max_norm,
                      bound=getattr(args, 'pivrg_bound', 2.0),
                      mintemp=getattr(args, 'pivrg_mintemp', 10)),
            consmtl=dict(max_norm=args.max_norm,
                        lambda_=getattr(args, 'consmtl_lambda', 1.0)),
            consfamo=dict(max_norm=args.max_norm,
                        lambda_=getattr(args, 'consmtl_lambda', 1.0),
                        gamma=getattr(args, 'gamma', 0.01),
                        w_lr=getattr(args, 'method_params_lr', 0.025)),
            conscityfamo=dict(max_norm=args.max_norm,
                        lambda_=getattr(args, 'consmtl_lambda', 1.0),
                        gamma=getattr(args, 'gamma', 0.01),
                        w_lr=getattr(args, 'method_params_lr', 0.025)),
            rbldc=dict(max_norm=args.max_norm,
                      lambda_param=getattr(args, 'rbldc_lambda', 0.1),
                      alpha=getattr(args, 'rbldc_alpha', 0.1),
                      feat_dim=getattr(args, 'rbldc_feat_dim', 512),
                      gamma=getattr(args, 'rbldc_gamma', 0.01),
                      w_lr=getattr(args, 'method_params_lr', 0.025)),
            go4align=dict(num_groups=args.num_groups,
                         robust_step_size=getattr(args, 'go4align_robust_step_size', 0.0001),
                         max_norm=args.max_norm),
            go4aligncov=dict(num_groups=args.num_groups,
                         robust_step_size=getattr(args, 'go4align_robust_step_size', 0.0001),
                         gamma3=getattr(args, 'go4align_gamma3', 0.1),
                         cov_history_size=getattr(args, 'go4align_history_size', 50),
                         max_norm=args.max_norm),
            grape=dict(K=getattr(args, 'grape_K', 2),
                     alpha=getattr(args, 'grape_alpha', 0.1),
                     tau=getattr(args, 'grape_tau', 1.0),
                     max_norm=args.max_norm),
            famo_lbfgs=dict(m=getattr(args, 'famo_lbfgs_m', 10),
                           w_lr=getattr(args, 'method_params_lr', 0.001),
                           max_norm=args.max_norm),
        )
    )
    return weight_methods_parameters
