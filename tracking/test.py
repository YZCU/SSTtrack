import os
import sys
import argparse
import torch

torch.set_num_threads(1)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


def run_tracker(tracker_name, tracker_param, save_name, run_id=None, dataset_name='otb', sequence=None, debug=0,
                threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, save_name, run_id)]
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main(num, dataset, cfg):
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='ssttrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default=cfg, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default=dataset, help='(otb, nfs, uav, tpl, vot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level: 0, 1')
    parser.add_argument('--save_name', type=str, default='ssttrack_hivitb_v1-' + str(num), help='Name of save file.')
    parser.add_argument('--threads', type=int, default=5, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    set_state(num, args.tracker_param)
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    run_tracker(args.tracker_name, args.tracker_param, args.save_name, args.runid, args.dataset_name, seq_name,
                args.debug,
                args.threads, num_gpus=args.num_gpus)


import yaml


def set_state(state, tracker_param):
    path = '../experiments/ssttrack/' + tracker_param + '.yaml'
    yaml_dir = os.path.join(os.path.dirname(__file__), path)
    with open(yaml_dir, 'r') as f:
        doc = yaml.safe_load(f)
    doc['TEST']['EPOCH'] = state
    with open(path, 'w') as f:
        yaml.dump(doc, f)


if __name__ == '__main__':
    cfg = 'ssttrack-ep150-full-256'
    datasets = ['otb']
    epoch = 22
    for dataset in datasets:
        main(epoch, dataset, cfg)
