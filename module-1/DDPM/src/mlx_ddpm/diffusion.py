from dataclasses import dataclass

import mlx.core as mx


def _extract(values: mx.array, timesteps: mx.array, x_shape: tuple[int, ...]) -> mx.array:
    return values[timesteps].reshape((timesteps.shape[0],) + (1,) * (len(x_shape) - 1))


@dataclass
class DDPMSchedule:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        self.betas = mx.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = mx.concatenate([mx.ones((1,)), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = mx.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

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

    def sample(self, model, shape: tuple[int, int, int, int]) -> mx.array:
        x = mx.random.normal(shape)
        for step in reversed(range(self.timesteps)):
            t = mx.full((shape[0],), step, dtype=mx.int32)
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
        return mx.clip(x, -1.0, 1.0)
