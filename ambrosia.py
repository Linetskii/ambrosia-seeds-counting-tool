"""Entry point"""

from watershed import Sample
from visualisation import Visualization
import argparse


def main():
    parser = argparse.ArgumentParser(description='Calculate the ambrosia seeds on photo. '
                                                 'Double right click to exclude, '
                                                 'and double left click to add seed',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", help="add file")
    args = parser.parse_args()
    path = args.file
    seeds = Sample(path)
    Visualization(*seeds.count()).show()


if __name__ == '__main__':
    main()
