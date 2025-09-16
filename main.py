from PIG.config import PIGConfig
from PIG.utils.multi_diffusion import MultiDiffusion

if __name__ == "__main__":
    config = PIGConfig()
    diffusion = MultiDiffusion(config)
    diffusion.train()
    # diffusion.sample()

