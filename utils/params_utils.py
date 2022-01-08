import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, help='Batch size', type=int)
    parser.add_argument("--emb_size", default=128, help="Embedding size", type=int)
    parser.add_argument("--hidden_size", default=256, help="Hidden size", type=int)
    parser.add_argument("--epochs", default=10, help="Training epochs", type=int)
    parser.add_argument("--lr", default=1e-3, help="Learning rate", type=float)
    parser.add_argument("--num_layers", default=2, help="Num layers of encoder GRU module", type=int)
    parser.add_argument("--dropout", default=0.5, help="Dropout ratio", type=float)

    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == '__main__':
    params = get_params()
    print(params)