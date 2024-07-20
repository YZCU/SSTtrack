from lib.test.analysis.plot_results import print_results, plot_results
from lib.test.evaluation import get_dataset, trackerlist
import os
trackers = []
dataset_name = 'otb'
file_name = 'ssttrack-ep150-full-256'
rect_path = 'output/results/ssttrack/' + file_name + '/otb'
rect_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', rect_path))
folders = [name for name in os.listdir(rect_path) if os.path.isdir(os.path.join(rect_path, name))]
for name in folders:
    trackers.extend(trackerlist(name='ssttrack', parameter_name=file_name, dataset_name=dataset_name,
                                save_name=name, run_ids=None, display_name=name))
dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
