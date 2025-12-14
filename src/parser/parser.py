from argparse import ArgumentParser

def parser_args():
    parser = ArgumentParser(description="train JEPA model")
    parser.add_argument("--depth", help="depth of ViT model (default: 6)", default=6)
    parser.add_argument("--drop_rate", help="drop rate of ViT model (default: 0.1)", default=0.1)
    parser.add_argument("--channel", help="num of color channels of the input image (default: 3)", default=3)
    parser.add_argument("--image_size", help="size of input image (default: 128)", default=128)
    parser.add_argument("--embed_dim", help="embedding dimension for the ViT model (default: 256)", default=256)
    parser.add_argument("--mlp_dim", help="MLP dimension used in ViT (default: 512)", default=512)
    parser.add_argument("--epochs", help="num of iteration when training the JEPA model (default: 20)", default=20)
    parser.add_argument("--num_heads", help="num of attention heads in the ViT (default: 8)", default=8)
    parser.add_argument("--batch_size", help="batch size when processing images (default: 16)", default=16)
    parser.add_argument("--lr", help="learning rate used in training (default: 0.001)", default=0.001)
    parser.add_argument("--momentum", help="momentum rate for EMA update (default: 0.996)", default=0.996)
    parser.add_argument("--num_target", help="number of target blocks used in I-JEPA (default: 4)", default=4)
    parser.add_argument("--num_classes", help="num of classes in the dataset (default: 10)", default=10)
    parser.add_argument("--sentence_encoder", help="sentence encoder model name for multimodal JEPA (default: all-MiniLM-L6-v2)", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--multimodal_run", help="run model in multimodal mode (default: y)", default="y")
    parser.add_argument("--patch_size", help="patch size in pixels used when processing images (default: 16)", default=16)
    parser.add_argument("--debug", help="run training in debug mode (default: y)", default="y")

    args = parser.parse_args()

    return args