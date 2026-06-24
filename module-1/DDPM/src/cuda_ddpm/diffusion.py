from dataclasses import dataclass

import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    return values[timesteps].reshape((timesteps.shape[0],) + (1,) * (len(x_shape) - 1))


@dataclass
class DDPMSchedule:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: torch.device | str = "cuda"

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones((1,), device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape) * noise
        )

    def training_loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, timesteps, noise)
        pred_noise = model(xt, timesteps)
        return torch.mean((noise - pred_noise) ** 2)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, int, int, int]) -> torch.Tensor:
        was_training = model.training
        model.eval()
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            beta_t = _extract(self.betas, t, x.shape)
            sqrt_one_minus_alpha_bar_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, t, x.shape)
            mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)

            if step > 0:
                variance = _extract(self.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(variance) * torch.randn_like(x)
            else:
                x = mean
        if was_training:
            model.train()
        return torch.clamp(x, -1.0, 1.0)
