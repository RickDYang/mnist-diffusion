import os

from tqdm import tqdm
import torch
from torchvision.utils import save_image

from huggingface_hub import upload_file, create_repo, hf_hub_download

from diffusion import DDPM


# pylint: disable=not-callable
class MnistDiffusionModel:
    def __init__(self, n_times=1000):
        # Basic information of Mnist dataset
        self.image_resolution = (1, 28, 28)
        self.n_classes = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DDPM(
            self.image_resolution,
            self.n_classes,
            embedding_layers=(256, 256, 512),
            # the max size of hidden_channels determined
            # by the size of the image resolution which
            # how many 2 factor can be divided by the image resolution
            # for example: 28 = 2*2*7, so the max size of hidden_channels is 2+1
            hidden_channels=(128, 256, 512),
            kernel_size=3,
            betas=(1e-4, 0.02),
            n_times=n_times,
            device=self.device,
        )
        self.model = self._init_parameters(model)
        self.model_fn = "mnist_diffusion_model.pt"
        self.hf_modle_fn = "mnist_diffusion"

    def _init_parameters(self, model):
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return model.to(self.device)

    def load(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        self.model.load_state_dict(torch.load(model_fn))

    def train(self, train_loader, validate_loader, epochs: int = 10, lr: float = 1e-4):
        # optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Train the model in one epoch
            train_loss = self.train_epoch(
                train_loader, f"Epoch {epoch+1}/{epochs} Train", optimizer, criterion
            )

            val_loss = self.evaluate(
                validate_loader, f"Epoch {epoch+1}/{epochs} Eval", criterion
            )

            # save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_fn)

            print(
                f"Epoch {epoch+1}/{epochs}, Train loss {train_loss}, Val loss {val_loss}"
            )

    def train_epoch(self, train_loader, desc, optimizer, criterion):
        self.model.train()
        train_loss = 0
        total = len(train_loader.dataset)
        for data, digits in tqdm(train_loader, desc=desc):
            data = data.to(self.device)
            digits = digits.to(self.device)

            optimizer.zero_grad()
            _, epsilon, pred_epsilon = self.model(data, digits)

            # pylint: disable=line-too-long
            # epsilon is:
            # Math: \epsilon_t
            # pred_epsilon is:
            # Math: \epsilon_\theta(X_t, t)
            # So the loss(MSE) is:
            # Math: ||\epsilon_t - \epsilon_\theta(X_t, t) ||^2 = ||\epsilon_t - \epsilon_\theta(\sqrt{\overline\alpha_t} * X_0 + \sqrt{1-\overline\alpha_t} * \epsilon, t)||^2
            # While:
            # Math: X_t = \sqrt{\overline\alpha_t} * X_0 + \sqrt{1-\overline\alpha_t} * \epsilon
            loss = criterion(pred_epsilon, epsilon)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        return train_loss / total

    def evaluate(self, validate_loader, desc, criterion):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, digits in tqdm(validate_loader, desc=desc):
                data = data.to(self.device)
                digits = digits.to(self.device)
                _, epsilon, pred_epsilon = self.model(data, digits)
                val_loss += criterion(pred_epsilon, epsilon).item()

        return val_loss / len(validate_loader.dataset)

    def infer(self, promotes: list[int]):
        self.model.eval()
        images = []
        with torch.no_grad():
            promote_tensors = torch.tensor(promotes).to(self.device)
            digit_images = self.model.decode(promote_tensors)

            # save the image to the test folder
            if not os.path.exists("test"):
                os.makedirs("test")
            for i, promote in enumerate(promotes):
                image = digit_images[i].cpu().view(*self.image_resolution)
                save_image(image, f"test/mnist_diffusion_sample_{promote}.png")
                images.append(image)
        return images

    def upload(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        token = os.getenv("HUGGINGFACE_TOKEN")
        repo_id = os.getenv("HUGGINGFACE_REPO")
        create_repo(
            repo_id,
            token=token,
            private=False,
            repo_type="model",
            exist_ok=True,
        )

        upload_file(
            repo_id=repo_id,
            path_or_fileobj=model_fn,
            path_in_repo=self.hf_modle_fn,
            token=token,
        )

    def from_pretrain(self):
        repo_id = os.getenv("HUGGINGFACE_REPO")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=self.hf_modle_fn,
            cache_dir="./cache",
        )
        # model_path = try_to_load_from_cache(repo_id=repo_id, filename=self.hf_modle_fn)
        self.load(model_path)
