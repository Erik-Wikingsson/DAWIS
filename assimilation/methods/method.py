class AssimilationMethod:
    """
    Base class for data assimilation methods.
    """

    def __init__(self, args):
        self.args = args

    def assimilate(self, x_forecast, obs, **kwargs):
        """
        Perform a data assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim). In scalefact units.
            obs: Observations, shape (obs_dim,).
            **kwargs: Additional keyword arguments that can be passed to specific implementations.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). In scalefact units.
        """
        raise NotImplementedError("Assimilation method not implemented.")
