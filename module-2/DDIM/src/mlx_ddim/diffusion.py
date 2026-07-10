from dataclasses import dataclass
import math

import mlx.core as mx


def _extract(values: mx.array, timesteps: mx.array, x_shape: tuple[int, ...]) -> mx.array:
    return values[timesteps].reshape((timesteps.shape[0],) + (1,) * (len(x_shape) - 1))


@dataclass
class DDIMSchedule:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"

    def __post_init__(self):
        if self.schedule == "linear":
            self.betas = mx.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.schedule == "cosine":
            self.betas = self._cosine_betas()
        else:
            raise ValueError(f"unknown noise schedule: {self.schedule}")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = mx.concatenate([mx.ones((1,)), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = mx.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _cosine_betas(self) -> mx.array:
        steps = self.timesteps + 1
        s = 0.008
        x = mx.linspace(0, self.timesteps, steps)
        alphas_cumprod = mx.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return mx.clip(betas, 1e-8, 0.999)

    def add_noise(self, x0: mx.array, timesteps: mx.array, noise: mx.array) -> mx.array:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape) * noise
        )

    def training_loss(self, model, x0: mx.array) -> mx.array:
        batch = x0.shape[0]
        timesteps = mx.random.randint(0, self.timesteps, (batch,))
        noise = mx.random.normal(x0.shape)
        xt = self.add_noise(x0, timesteps, noise)
        pred_noise = model(xt, timesteps)
        return mx.mean((noise - pred_noise) ** 2)

    def sample(
        self,
        model,
        shape: tuple[int, int, int, int],
        sampler: str = "ddim",
        sample_steps: int | None = None,
        eta: float = 0.0,
    ) -> mx.array:
        x = mx.random.normal(shape)
        if sampler == "ancestral":
            x = self._sample_ancestral(model, x)
        elif sampler == "ddim":
            x = self._sample_ddim(model, x, sample_steps or self.timesteps, eta)
        else:
            raise ValueError(f"unknown sampler: {sampler}")
        return mx.clip(x, -1.0, 1.0)

    def _sample_ancestral(self, model, x: mx.array) -> mx.array:
        for step in reversed(range(self.timesteps)):
            t = mx.full((x.shape[0],), step, dtype=mx.int32)
            pred_noise = model(x, t)
            beta_t = _extract(self.betas, t, x.shape)
            sqrt_one_minus_alpha_bar_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, t, x.shape)
            mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)

            if step > 0:
                variance = _extract(self.posterior_variance, t, x.shape)
                x = mean + mx.sqrt(variance) * mx.random.normal(x.shape)
            else:
                x = mean
            mx.eval(x)
        return x

    def _sample_ddim(self, model, x: mx.array, sample_steps: int, eta: float) -> mx.array:
        if sample_steps < 1 or sample_steps > self.timesteps:
            raise ValueError("sample_steps must be between 1 and timesteps")

        if sample_steps == 1:
            steps = [self.timesteps - 1]
        else:
            steps = [round(i * (self.timesteps - 1) / (sample_steps - 1)) for i in range(sample_steps)]
        steps = list(dict.fromkeys(steps))
        for index in reversed(range(len(steps))):
            step = steps[index]
            prev_step = steps[index - 1] if index > 0 else -1
            t = mx.full((x.shape[0],), step, dtype=mx.int32)
            pred_noise = model(x, t)

            alpha_t = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else mx.array(1.0)
            pred_x0 = (x - mx.sqrt(1.0 - alpha_t) * pred_noise) / mx.sqrt(alpha_t)
            pred_x0 = mx.clip(pred_x0, -1.0, 1.0)

            sigma = eta * mx.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
            direction_scale = mx.sqrt(mx.maximum(1.0 - alpha_prev - sigma**2, 0.0))
            x = mx.sqrt(alpha_prev) * pred_x0 + direction_scale * pred_noise
            if eta > 0 and prev_step >= 0:
                x = x + sigma * mx.random.normal(x.shape)
            mx.eval(x)
        return x
