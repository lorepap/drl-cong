import sys
import argparse
from typing import Any
from helper import utils


def verify_models(models) -> list:
    models = models.split()
    yaml_config = utils.parse_models_config()
    all_models = yaml_config['models'].keys()

    for ml in models:
        if ml not in all_models:
            sys.exit('%s is not a model included in config/models.yml' % ml)

    return all_models

def verify_reward(rewards) -> list:
    rewards = rewards.split()
    yaml_config = utils.parse_reward_config()
    all_rewards = yaml_config['rewards'].keys()

    for r in rewards:
        if r not in all_rewards:
            sys.exit('%s is not a reward included in config/reward.yml' % r)

    return all_rewards


def verify_protocols(protocols) -> list:
    protocols = protocols.split()
    yaml_config = utils.parse_protocols_config()
    all_protocols = yaml_config['protocols'].keys()

    for ml in protocols:
        if ml not in all_protocols:
            sys.exit(
                '%s is not a protocol included in config/protocols.yml' % ml)

    return all_protocols


def verify_pantheon_protocols(protocols) -> list:
    protocols = protocols.split()
    yaml_config = utils.parse_pantheon_protocols_config()
    all_protocols = yaml_config['schemes'].keys()

    for sm in protocols:
        if sm not in all_protocols:
            sys.exit(
                '%s is not part of the protocols implemented in patheon' % sm)

    return all_protocols


def verify_trace(trace: str) -> list:
    yaml_config = utils.parse_traces_config()
    all_traces = yaml_config['traces'].keys()
    print(all_traces)

    if trace not in all_traces:
        sys.exit(
            '%s is not a trace included in config/traces.yml' % trace)

    return all_traces


def add_base_arguments(parser: argparse.ArgumentParser) -> None:

    parser.add_argument('--trace', '-t', type=str,
                        help='--trace: Name of mahimahi trace file used', default="none")

    parser.add_argument('--ip', '-x', type=str,
                        help='--ip: IP of iperf server machine', default="10.120.8.116")

    parser.add_argument('--time', '-e', type=int,
                        help='--time: Number of seconds to run iperf', default=86400)

    parser.add_argument('--iperf', '-u', type=int,
                        help='--iperf: Flag (0:false, 1:true) to indicate whether to use iperf or not', default=1)

    parser.add_argument('--iperf_dir', '-d', type=str,
                        help='--iperf_dir: iperf directory to use', default="log/iperf")

    parser.add_argument('--trace_dir', '-f', type=str,
                        help='--trace_dir: trace directory to use', default="log/prod")

    parser.add_argument('--runs', '-r', type=int,
                        help='--runs: Number of times to run', default=1)
    
    parser.add_argument('--reward', '-rw', type=str, help='Name of the reward for the training',
                        default='orca')

    # single protocol mode
    parser.add_argument("--protocol", "-prot", help="Native protocol under test within mimic", default="cubic")

    parser.add_argument('--model_name', '-mN', help="String stored in the log directory of the specific trained model", default=None)


def parse_train_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/train.py --all')

    # schemes related
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--all', '-a', action='store_true',
                       help='Run all models')

    group.add_argument('--models', '-m', metavar='"MODEL1 MODEL2..."',
                       help='Run only space-separated list of models', default='active_explorer')

    parser.add_argument('--retrain', '-rt', type=int,
                        help='--retrain: Flag (0:false, 1:true) to retrain latest model or not', default=1)

    add_base_arguments(parser)

    args = parser.parse_args()

    if not args.all and not args.models:
        sys.exit('Must specify --all or --models')

    if args.models is not None:
        verify_models(args.models)

    if args.reward is not None:
        verify_reward(args.reward)  

    verify_trace(args.trace)

    return args


def parse_pantheon_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 experiment/pantheon.py --test')

    
    # schemes related
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--all', '-a', action='store_true', help='Run tests')

    group.add_argument('--protocols', '-p', metavar='"PROTOCOL1 PROTOCOL2..."',
                       help='Run only space-separated list of protocols')

    add_base_arguments(parser)

    args = parser.parse_args()

    if not args.all and not args.protocols:
        sys.exit('Must specify --all or --protocols')

    if args.protocols is not None:
        verify_pantheon_protocols(args.protocols)

    
    verify_trace(args.trace)

    return args


def parse_harm_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/harm.py')

    group = parser.add_mutually_exclusive_group()
    
    group.add_argument('--solo', action='store_true', help='Run only cubic')
    group.add_argument('--against', action='store_true', help='Run cubic against mimic')

    add_base_arguments(parser)

    args = parser.parse_args()

    if not args.solo and not args.against:
        sys.exit('Must specify --solo or --against')
                

    verify_trace(args.trace)

    return args




def parse_predict_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/predict.py --all')

    # schemes related
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--all', '-a', action='store_true',
                       help='Run all models')

    group.add_argument('--models', '-m', metavar='"MODEL1 MODEL2..."',
                       help='Run only space-separated list of models')

    add_base_arguments(parser)

    args = parser.parse_args()

    if not args.all and not args.models:
        sys.exit('Must specify --all or --models')

    if args.models is not None:
        verify_models(args.models)

    
    verify_trace(args.trace)

    return args


def parse_collect_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/collect.py cubic 1000')

    # schemes related
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--all', '-a', action='store_true',
                       help='Run all protocols')

    group.add_argument('--protocols', '-p', metavar='"PROTOCOL1 PROTOCOL2..."',
                       help='Run only space-separated list of protocols')

    parser.add_argument('--logsize', '-s', type=int,
                        help='--ls: number of logs to collect', default=1000)

    add_base_arguments(parser)


    args = parser.parse_args()

    if not args.all and not args.protocols:
        sys.exit('Must specify --all or --protocols')

    if args.protocols is not None:
        verify_protocols(args.protocols)

    
    verify_trace(args.trace)

    return args


def parse_basic_setup() -> Any:
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/basic.py')

    add_base_arguments(parser)

    args = parser.parse_args()

    verify_trace(args.trace)

    return args


def parse_test_setup():
    parser = argparse.ArgumentParser(
        description='by default, run python3 ml/test.py --all')

    # schemes related
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--all', '-a', action='store_true',
                       help='run test mode')

    group.add_argument('--models', '-m', metavar='"MODEL1 MODEL2..."',
                       help='set up a space-separated list of models', default="bootstrapped_ucb")

    add_base_arguments(parser)


    args = parser.parse_args()

    if not args.all and not args.models:
        sys.exit('Must specify --all or --models')

    if args.models is not None:
        verify_models(args.models)

    
    verify_trace(args.trace)

    return args
