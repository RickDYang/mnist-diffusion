import argparse
from dotenv import load_dotenv

from data import load_data
from model import MnistDiffusionModel


def train():
    # The batch size of data in training
    batch_size = 256
    # The traning epochs, for my tesing, 100 is a good epochs
    epochs = 100
    # user all train data to train, and validate on validate data
    train_loader, _, val_loader = load_data(batch_size, ratio=1)
    # Init and create the model
    model = MnistDiffusionModel()
    # model.load("mnist_diffusion_model_best.pt")
    # train and validate in the epochs
    model.train(train_loader, val_loader, epochs)
    return model


def infer():
    # Load the model from the saved weights
    model = MnistDiffusionModel()
    # model.load("mnist_diffusion_model_best.pt")
    model.load()
    model.infer(list(range(10)))
    return model


def upload():
    # Init and create the model
    model = MnistDiffusionModel()
    model.upload()


def from_pretrain():
    # Init and create the model
    model = MnistDiffusionModel()
    model.from_pretrain()
    model.infer(list(range(10)))


# The major progress include train, infer, upload, and from_pretrain
# - During train, the model will be saved to "mnist_diffusion_model.pt" for the best model
# - In the inferencing, the model will be loaded from "mnist_diffusion_model.pt"
# - You can use upload to upload the model to huggingface hub
# - You can use from_pretrain to download the model from huggingface hub and make inference
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run MNIST generation model with different modes: train/infer/upload/from_pretrain"
    )
    parser.add_argument("mode", choices=["train", "infer", "upload", "from_pretrain"], help="The mode to run")
    args = parser.parse_args()

    if args.mode == "train":
        train()
        infer()
    elif args.mode == "infer":
        infer()
    elif args.mode == "upload":
        upload()
    elif args.mode == "from_pretrain":
        from_pretrain()
    else:
        print(f"The mode={args.mode} not supported")
