import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Task import Task
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', help='train or predict')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    trainer = Task()
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'predict':
        trainer.predict()
    else:
        raise ValueError('mode should be train or predict')
