# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOM_TYPES = {
    'conferenceRoom': 0,
    'copyRoom': 1,
    'hallway': 2,
    'office': 3,
    'pantry': 4,
    'WC': 5,
    'auditorium': 6,
    'storage': 7,
    'lounge': 8,
    'lobby': 9,
    'openspace': 10,
}

INV_OBJECT_LABEL = {
    0: 'ceiling',
    1: 'floor',
    2: 'wall',
    3: 'beam',
    4: 'column',
    5: 'window',
    6: 'door',
    7: 'chair',
    8: 'table',
    9: 'bookcase',
    10: 'sofa',
    11: 'board',
    12: 'clutter',
}

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


def object_name_to_label(object_class):
    r"""convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL['clutter'])
    return object_label


# modify from https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/s3dis.py  # noqa
def read_s3dis_format(area_id: str,
                      room_name: str,
                      data_root: str = './',
                      label_out: bool = True,
                      verbose: bool = False):
    r"""
    extract data from a room folder
    """
    room_type = room_name.split('_')[0]
    room_label = ROOM_TYPES[room_type]
    room_dir = osp.join(data_root, area_id, room_name)  # s3dis/Area_1/conferenceRoom_1
    raw_path = osp.join(room_dir, f'{room_name}.txt')   # s3dis/Area_1/conferenceRoom_1/conferenceRoom_1.txt

    # 先用pandas读取，方便数据清洗
    df = pd.read_csv(raw_path, sep=' ', header=None, dtype=str)
    # 尝试将前三列和3~5列转为float，无法转的行直接删除
    for col in range(6):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[0, 1, 2, 3, 4, 5])  # 删除xyz和rgb有异常的行
    room_ver = df.values
    xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype='float32')
    rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype='uint8')
    if not label_out:
        return xyz, rgb
    n_ver = len(room_ver)
    del room_ver
    del df
    # 建立原始点云的KD树
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    semantic_labels = np.zeros((n_ver, ), dtype='int64')
    room_label = np.asarray([room_label])
    instance_labels = np.ones((n_ver, ), dtype='int64') * -100
    objects = glob.glob(osp.join(room_dir, 'Annotations', '*.txt')) # s3dis/Area_1/conferenceRoom_1/Annotations/*.txt
    i_object = 1
    for single_object in objects:   # s3dis/Area_1/conferenceRoom_1/Annotations/beam_1.txt
        object_name = os.path.splitext(os.path.basename(single_object))[0]  # 'beam_1'
        if verbose:
            print(f'adding object {i_object} : {object_name}')
        object_class = object_name.split('_')[0]    # beam
        object_label = object_name_to_label(object_class)   # 3
        # 对每个物体点云，找到它们在原始点云中最接近的点
        obj_ver = pd.read_csv(single_object, sep=' ', header=None).values
        _, obj_ind = nn.kneighbors(obj_ver[:, 0:3]) # 返回的 obj_ind 是原始房间点云 xyz 中的索引
        
        semantic_labels[obj_ind] = object_label
        instance_labels[obj_ind] = i_object
        i_object = i_object + 1

    return (
        xyz,
        rgb,
        semantic_labels,
        instance_labels,    # 每个room的instance 进行标注，不是所有的一起标注
        room_label,
    )


def get_parser():
    parser = argparse.ArgumentParser(description='s3dis data prepare')
    parser.add_argument(
        '--data-root', type=str, default='./Stanford3dDataset_v1.2_Aligned_Version', help='root dir save data')
    parser.add_argument(
        '--save-dir', type=str, default='./preprocess', help='directory save processed data')
    parser.add_argument(
        '--patch', action='store_true', help='patch data or not (just patch at first time running)')
    parser.add_argument('--align', action='store_true', help='processing aligned dataset or not')
    parser.add_argument('--verbose', action='store_true', help='show processing room name or not')

    args_cfg = parser.parse_args()

    return args_cfg


# patch -ruN -p0 -d  raw < s3dis.patch
if __name__ == '__main__':
    args = get_parser()
    data_root = args.data_root
    # processed data output dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.patch:
        """
        如果同时指定了 --patch 和 --align 命令行参数，它会将 s3dis_align.patch 文件中定义的更改应用到 data_root 目录下的已对齐的 S3DIS 数据集文件。
        """
        if args.align:  # processing aligned s3dis dataset
            os.system(
                f"patch -ruN -p0 -d  {data_root} < {osp.join(osp.dirname(__file__), 's3dis_align.patch')}"  # noqa
            )
            # rename to avoid room_name conflict
            if osp.exists(osp.join(data_root, 'Area_6', 'copyRoom_1', 'copy_Room_1.txt')):
                os.rename(
                    osp.join(data_root, 'Area_6', 'copyRoom_1', 'copy_Room_1.txt'),
                    osp.join(data_root, 'Area_6', 'copyRoom_1', 'copyRoom_1.txt'))
        else:
            os.system(
                f"patch -ruN -p0 -d  {data_root} < {osp.join(osp.dirname(__file__), 's3dis.patch')}"
            )

    area_list = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    # area_list = ['Area_1']
    # area_list = ['Area_2']
    # area_list = ['Area_3']
    # area_list = ['Area_4']
    # area_list = ['Area_5']
    # area_list = ['Area_6']

    for area_id in area_list:
        print(f'Processing: {area_id}')
        area_dir = osp.join(data_root, area_id)
        # get the room name list for each area
        room_name_list = os.listdir(area_dir)
        try:
            room_name_list.remove(f'{area_id}_alignmentAngle.txt')
            room_name_list.remove('.DS_Store')
        except:  # noqa
            pass
        for room_name in room_name_list:
            # 跳过不符合命名格式的文件/目录
            if room_name.startswith('.') or '_' not in room_name:
                # if args.verbose:
                print(f"跳过无效的房间名称: {room_name}")
                continue
                
            scene = f'{area_id}_{room_name}'
            if args.verbose:
                print(f'processing: {scene}')
            save_path = osp.join(save_dir, scene + '_inst_nostuff.pth')
            if osp.exists(save_path):
                continue
            (xyz, rgb, semantic_labels, instance_labels,
             room_label) = read_s3dis_format(area_id, room_name, data_root)
            rgb = (rgb / 127.5) - 1     # normalize to [-1, 1]
            torch.save((xyz, rgb, semantic_labels, instance_labels, room_label, scene), save_path)
