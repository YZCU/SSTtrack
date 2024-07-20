import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class OTBDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "ball", "path": "ball", "startFrame": 1, "endFrame": 625, "nz": 4, "ext": "png",
             "anno_path": "ball/groundtruth_rect.txt", "object_class": "person"},
            {"name": "basketball", "path": "basketball", "startFrame": 1, "endFrame": 186, "nz": 4, "ext": "png",
             "anno_path": "basketball/groundtruth_rect.txt", "object_class": "person"},
            {"name": "board", "path": "board", "startFrame": 1, "endFrame": 471, "nz": 4, "ext": "png",
             "anno_path": "board/groundtruth_rect.txt", "object_class": "person"},
            {"name": "book", "path": "book", "startFrame": 1, "endFrame": 601, "nz": 4, "ext": "png",
             "anno_path": "book/groundtruth_rect.txt", "object_class": "person"},
            {"name": "bus", "path": "bus", "startFrame": 1, "endFrame": 131, "nz": 4, "ext": "png",
             "anno_path": "bus/groundtruth_rect.txt", "object_class": "person"},
            {"name": "bus2", "path": "bus2", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "png",
             "anno_path": "bus2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "campus", "path": "campus", "startFrame": 1, "endFrame": 976, "nz": 4, "ext": "png",
             "anno_path": "campus/groundtruth_rect.txt", "object_class": "person"},
            {"name": "car", "path": "car", "startFrame": 1, "endFrame": 101, "nz": 4, "ext": "png",
             "anno_path": "car/groundtruth_rect.txt", "object_class": "person"},
            {"name": "car2", "path": "car2", "startFrame": 1, "endFrame": 131, "nz": 4, "ext": "png",
             "anno_path": "car2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "car3", "path": "car3", "startFrame": 1, "endFrame": 331, "nz": 4, "ext": "png",
             "anno_path": "car3/groundtruth_rect.txt", "object_class": "person"},
            {"name": "card", "path": "card", "startFrame": 1, "endFrame": 930, "nz": 4, "ext": "png",
             "anno_path": "card/groundtruth_rect.txt", "object_class": "person"},
            {"name": "coin", "path": "coin", "startFrame": 1, "endFrame": 149, "nz": 4, "ext": "png",
             "anno_path": "coin/groundtruth_rect.txt", "object_class": "person"},
            {"name": "coke", "path": "coke", "startFrame": 1, "endFrame": 731, "nz": 4, "ext": "png",
             "anno_path": "coke/groundtruth_rect.txt", "object_class": "person"},
            {"name": "drive", "path": "drive", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "png",
             "anno_path": "drive/groundtruth_rect.txt", "object_class": "person"},
            {"name": "excavator", "path": "excavator", "startFrame": 1, "endFrame": 501, "nz": 4, "ext": "png",
             "anno_path": "excavator/groundtruth_rect.txt", "object_class": "person"},
            {"name": "face", "path": "face", "startFrame": 1, "endFrame": 279, "nz": 4, "ext": "png",
             "anno_path": "face/groundtruth_rect.txt", "object_class": "person"},
            {"name": "face2", "path": "face2", "startFrame": 1, "endFrame": 1111, "nz": 4, "ext": "png",
             "anno_path": "face2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "forest", "path": "forest", "startFrame": 1, "endFrame": 530, "nz": 4, "ext": "png",
             "anno_path": "forest/groundtruth_rect.txt", "object_class": "person"},
            {"name": "forest2", "path": "forest2", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "png",
             "anno_path": "forest2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "fruit", "path": "fruit", "startFrame": 1, "endFrame": 552, "nz": 4, "ext": "png",
             "anno_path": "fruit/groundtruth_rect.txt", "object_class": "person"},
            {"name": "hand", "path": "hand", "startFrame": 1, "endFrame": 184, "nz": 4, "ext": "png",
             "anno_path": "hand/groundtruth_rect.txt", "object_class": "person"},
            {"name": "kangaroo", "path": "kangaroo", "startFrame": 1, "endFrame": 117, "nz": 4, "ext": "png",
             "anno_path": "kangaroo/groundtruth_rect.txt", "object_class": "person"},
            {"name": "paper", "path": "paper", "startFrame": 1, "endFrame": 278, "nz": 4, "ext": "png",
             "anno_path": "paper/groundtruth_rect.txt", "object_class": "person"},
            {"name": "pedestrain", "path": "pedestrain", "startFrame": 1, "endFrame": 306, "nz": 4, "ext": "png",
             "anno_path": "pedestrain/groundtruth_rect.txt", "object_class": "person"},
            {"name": "pedestrian2", "path": "pedestrian2", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "png",
             "anno_path": "pedestrian2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "player", "path": "player", "startFrame": 1, "endFrame": 901, "nz": 4, "ext": "png",
             "anno_path": "player/groundtruth_rect.txt", "object_class": "person"},
            {"name": "playground", "path": "playground", "startFrame": 1, "endFrame": 800, "nz": 4, "ext": "png",
             "anno_path": "playground/groundtruth_rect.txt", "object_class": "person"},
            {"name": "rider1", "path": "rider1", "startFrame": 1, "endFrame": 336, "nz": 4, "ext": "png",
             "anno_path": "rider1/groundtruth_rect.txt", "object_class": "person"},
            {"name": "rider2", "path": "rider2", "startFrame": 1, "endFrame": 210, "nz": 4, "ext": "png",
             "anno_path": "rider2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "rubik", "path": "rubik", "startFrame": 1, "endFrame": 526, "nz": 4, "ext": "png",
             "anno_path": "rubik/groundtruth_rect.txt", "object_class": "person"},
            {"name": "student", "path": "student", "startFrame": 1, "endFrame": 396, "nz": 4, "ext": "png",
             "anno_path": "student/groundtruth_rect.txt", "object_class": "person"},
            {"name": "toy1", "path": "toy1", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "png",
             "anno_path": "toy1/groundtruth_rect.txt", "object_class": "person"},
            {"name": "toy2", "path": "toy2", "startFrame": 1, "endFrame": 601, "nz": 4, "ext": "png",
             "anno_path": "toy2/groundtruth_rect.txt", "object_class": "person"},
            {"name": "trucker", "path": "trucker", "startFrame": 1, "endFrame": 221, "nz": 4, "ext": "png",
             "anno_path": "trucker/groundtruth_rect.txt", "object_class": "person"},
            {"name": "worker", "path": "worker", "startFrame": 1, "endFrame": 1209, "nz": 4, "ext": "png",
             "anno_path": "worker/groundtruth_rect.txt", "object_class": "person"}
        ]
        return sequence_info_list
