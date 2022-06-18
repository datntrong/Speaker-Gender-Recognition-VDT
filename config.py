import torch

RATE = '8000'

CATES = ['female', 'male']

BASE_ORIGINAL_TRAIN = '/gendersclassification-vdt-2022/dataset_v2/train/female/midside'
NUM_WORKERS = 2
BASE_TRAIN = BASE_ORIGINAL_TRAIN
INFER_ONLY = True # change this to False to train the model again
PATH_WAV_PUBLIC_TEST = '/gendersclassification-vdt-2022/public-test/public-test/wav'
PATH_WAV_PRIVATE_TEST = '/gendersclassification-vdt-2022/private-test/private-test/wav'
PATH_LABEL_PUBLIC_TEST = '../input/gendersclassification-vdt-2022/public-test/public-test/test_files.txt'


