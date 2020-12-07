import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--robot', default='baxter')
parser.add_argument('--config', default=None, required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--device', default='cuda')

if __name__ == '__main__':
    # TODO
    pass
