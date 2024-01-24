import torch
import hydra
import os
import wandb
from omegaconf import OmegaConf

# Import models



torch.manual_seed(42)


@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    # Insert model training code here


if __name__ == "__main__":
    main()