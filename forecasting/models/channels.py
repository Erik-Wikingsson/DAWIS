"""Channel counts for forecasting model inputs/outputs, derived from dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelBudget:
    vars: int      # variables per time step
    state: int     # channels of one state    (== vars)
    cond: int      # channels of the conditioning stack (vars * init_states)
    latent: int    # latent channels, for models that have a latent space

    @property
    def cond_plus_state(self) -> int:
        """Channels of the conditioning stack plus one state."""
        return self.cond + self.state


def channel_budget(md, args) -> ChannelBudget:
    n_vars = md.num_channels
    init_states = int(getattr(args, "init_states", 1) or 1)
    latent = getattr(args, "latent_channels", None)
    return ChannelBudget(
        vars=n_vars,
        state=n_vars,
        cond=n_vars * init_states,
        latent=int(latent) if latent else n_vars,
    )
