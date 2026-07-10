from dataclasses import dataclass
import math

import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    return values[timesteps].reshape((timesteps.shape[0],) + (1,) * (len(x_shape) - 1))


@dataclass
class DDIMSchedule:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"
    device: torch.device | str = "cuda"

    def __post_init__(self):
        self.device = torch.device(self.device)
        if self.schedule == "linear":
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        elif self.schedule == "cosine":
            self.betas = self._cosine_betas()
        else:
            raise ValueError(f"unknown noise schedule: {self.schedule}")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones((1,), device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _cosine_betas(self) -> torch.Tensor:
        steps = self.timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=1e-8, max=0.999)

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
    def sample(
        self,
        model,
        shape: tuple[int, int, int, int],
        sampler: str = "ddim",
        sample_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        x = torch.randn(shape, device=self.device)
        if sampler == "ancestral":
            x = self._sample_ancestral(model, x)
        elif sampler == "ddim":
            x = self._sample_ddim(model, x, sample_steps or self.timesteps, eta)
        else:
            raise ValueError(f"unknown sampler: {sampler}")
        if was_training:
            model.train()
        return torch.clamp(x, -1.0, 1.0)

    def _sample_ancestral(self, model, x: torch.Tensor) -> torch.Tensor:
        for step in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
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
        return x

    def _sample_ddim(self, model, x: torch.Tensor, sample_steps: int, eta: float) -> torch.Tensor:
        if sample_steps < 1 or sample_steps > self.timesteps:
            raise ValueError("sample_steps must be between 1 and timesteps")

        if sample_steps == 1:
            steps = torch.tensor([self.timesteps - 1], device=self.device, dtype=torch.long)
        else:
            steps = torch.linspace(0, self.timesteps - 1, sample_steps, device=self.device).round().long()
        steps = torch.unique_consecutive(steps)
        for index in reversed(range(len(steps))):
            step = int(steps[index].item())
            prev_step = int(steps[index - 1].item()) if index > 0 else -1
            t = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_t = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else torch.tensor(1.0, device=self.device)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
            direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0))
            x = torch.sqrt(alpha_prev) * pred_x0 + direction_scale * pred_noise
            if eta > 0 and prev_step >= 0:
                x = x + sigma * torch.randn_like(x)
        return x
