import os
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

from metrics import metrics
from plotting import plotting
from forecasting.models.ar_model import ARModel


class ARProbModel(ARModel):
    """
    Generic probabilistic auto-regressive forecasting model.
    Abstract class that can be extended.
    """

    def __init__(self, args):
        super().__init__(args)
        self.ensemble_size = args.ensemble_size

        self.val_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )
        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

        # TODO: Could be moved to ARModel base class
        self.spectra_metrics = {
            "spectra": [],  # For each time step and variable
            "spectra_gt": [],  # For each time step and variable
        }

    def sample_trajectories(
        self,
        init_states,
        num_traj,
        num_steps,
    ):
        """
        init_states: (B, N_steps, d_state, X, Y)
        num_traj: S, number of trajectories to sample

        Returns
        trajectories: (B, S, pred_steps, d_state, X, Y)
        """
        unroll_func = self.unroll_prediction

        traj_list = []
        for i in range(num_traj):
            traj = unroll_func(
                init_states,
                num_steps,
            )

            traj_list.append(traj)

        trajectories = torch.stack(traj_list, dim=1)

        return trajectories

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot ensemble forecast + mean and std
        """
        raise NotImplementedError(
            "plot_examples not implemented for ARProbModel")

    # TODO: Add plot_video function

    def ensemble_common_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Common step done during both evaluation and testing

        batch: tuple of tensors, batch to perform ensemble forecast on

        Returns:
        trajectories: (B, S, pred_steps, d_state, X, Y)
        target_states: (B, pred_steps, d_state, X, Y)
        spread_squared_batch: (B, pred_steps, d_state)
        ens_mse_batch: (B, pred_steps, d_state)
        """
        # Compute and store metrics for ensemble forecast
        init_states, target_states = batch

        trajectories = self.sample_trajectories(
            init_states,
            self.ensemble_size,
            target_states.shape[1],  # pred_steps
        )

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            mean_vars=False,
        )

        ens_mean = torch.mean(
            trajectories, dim=1
        )

        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            mean_vars=False,
        )

        return (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        super().validation_step(batch, batch_idx)
        # TODO: reuse an ensemble member instead of sampling an extra one
        init_states, true_states = batch
        prediction, loss = self.unroll_prediction_train(init_states, true_states,
                                                        self.args.ar_steps_eval)
        time_step_loss = torch.mean(loss, dim=0)  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            true_states,
            mean_vars=False,
        )
        self.val_metrics["mse"].append(entry_mses)

        # Get Probabilistic Metrics
        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)
        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )
        ens_std = torch.std(trajectories, dim=1)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            mean_vars=False,
        )
        self.val_metrics["ens_mae"].append(ens_maes)

        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            mean_vars=False,
        )
        self.val_metrics["crps_ens"].append(crps_batch)

        # Plot example predictions (on rank 0 only)
        if self.trainer.is_global_zero and batch_idx == 0:
            n_examples = 1  # Number of examples to plot
            init_states, target_states = batch

            # Rescale to original data scale
            traj_rescaled = ((trajectories * self.data_std) +
                             self.data_mean) * self.scalefact
            target_rescaled = ((target_states * self.data_std) +
                               self.data_mean) * self.scalefact

            # Compute mean and std of ensemble
            ens_mean = torch.mean(
                traj_rescaled, dim=1
            )
            ens_std = torch.std(
                traj_rescaled, dim=1
            )

            # Iterate over the examples
            for traj_slice, target_slice, ens_mean_slice, ens_std_slice in zip(
                traj_rescaled[:n_examples],
                target_rescaled[:n_examples],
                ens_mean[:n_examples],
                ens_std[:n_examples],
            ):
                # traj_slice ([n_steps, n_ens, d_state, X, Y])
                # target_slice ([n_steps, d_state, X, Y])

                # Iterate over prediction horizon time steps
                for t_i in range(1, traj_slice.shape[0] + 1):
                    # NOTE: val_steps_to_log are 1-indexed
                    if t_i in self.args.val_steps_to_log:
                        time_title_part = self.md.format_lead_time(t_i)
                        # Create one figure per variable at this time step
                        var_figs = [
                            plotting.plot_ensemble_prediction(
                                traj_slice[:, t_i-1, var_i].detach(
                                ).cpu().numpy(),
                                target_slice[t_i-1, var_i].detach(
                                ).cpu().numpy(),
                                ens_mean_slice[t_i-1,
                                               var_i].detach().cpu().numpy(),
                                ens_std_slice[t_i-1,
                                              var_i].detach().cpu().numpy(),
                                var_name=var_name,
                                title=f"{var_name}, {time_title_part}",
                                step=t_i,
                                # Fixed per-dataset colour scale
                                imshow_kwargs=self.md.imshow_kwargs(var_i),
                                spread_kwargs=self.md.imshow_kwargs(
                                    var_i, spread=True),
                            )
                            for var_i, var_name in enumerate(
                                self.md.variable_names
                            )
                        ]

                        example_title = f"example_{self.plotted_examples}_step_{t_i}"
                        wandb.log(
                            {
                                f"{var_name}_{example_title}": wandb.Image(fig)
                                for var_name, fig in zip(
                                    self.md.variable_names, var_figs
                                )
                            }
                        )
                        plt.close(
                            "all"
                        )  # Close all figs for this time step, saves memory

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        self.log_spsk_ratio(self.val_metrics, "val")

        # NOTE: This has to be called after logging spsk_ratio since it clears the stored metrics
        super().on_validation_epoch_end()

    def log_spsk_ratio(self, metric_vals, prefix):
        """
        Compute the mean spread-skill ratio for logging in evaluation

        metric_vals: dict with all metric values
        prefix: string, prefix to use for logging
        """
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))

            # Include finite sample correction
            spsk_ratios = np.sqrt(
                (self.ensemble_size + 1) / self.ensemble_size
            ) * (
                spread / skill
            )
            log_dict = self.create_metric_log_dict(
                spsk_ratios, prefix, "spsk_ratio"
            )

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(
                spsk_ratios
            )  # log mean
            wandb.log(log_dict)

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """
        super().test_step(batch, batch_idx)

        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_common_step(batch)
        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            mean_vars=False,
        )
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            mean_vars=False,
        )
        self.test_metrics["crps_ens"].append(crps_batch)

        # Spectra for each time step and variable
        # TODO: This can be done more efficiently
        # TODO: Move this to the base ARModel class
        target_spectra = metrics.power_spectrum(target_states.unsqueeze(1))

        radial_target_spectra = metrics.radial_average(target_spectra)
        self.spectra_metrics["spectra_gt"].append(
            radial_target_spectra)  # (B, t, n_freqs)

        traj_spectra = metrics.power_spectrum(trajectories)
        radial_traj_spectra = metrics.radial_average(traj_spectra)
        self.spectra_metrics["spectra"].append(
            radial_traj_spectra)  # (B, t, n_freqs)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
        ):
            # Rescale trajectories and target states to original data scale
            trajectories = (trajectories * self.data_std +
                            self.data_mean) * self.scalefact
            target_states = (target_states * self.data_std +
                             self.data_mean) * self.scalefact

            # Plot the video for the first example in the batch
            video = plotting.save_video(
                forecast=trajectories[0].transpose(0, 1).cpu().numpy(),
                truth=target_states[0].cpu().numpy(),
                oberrstdev=0,  # no observations in forecasting
                obs_p=0,
                obs_fn=lambda x: x,
                dir=wandb.run.dir,
                level=0,
                imshow_kwargs=self.md.imshow_kwargs(0),
                spread_kwargs=self.md.imshow_kwargs(0, spread=True),
            )

            wandb.log({
                "animation": wandb.Video(video, fps=10, format="mp4")
            })

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        super().on_test_epoch_end()
        self.log_spsk_ratio(self.test_metrics, "test")

        # TODO: Move this to the ARModel base class
        spectra_tensor = self.all_gather_cat(
            torch.cat(self.spectra_metrics["spectra"], dim=0).cpu()
        ).mean(dim=0).cpu()  # (pred_steps, d_state, R)

        spectra_gt_tensor = self.all_gather_cat(
            torch.cat(self.spectra_metrics["spectra_gt"], dim=0).cpu()
        ).mean(dim=0).cpu()  # (pred_steps, d_state, R)

        # TODO: Add log spectral distance metric over time steps
        # NOTE: Only logging spectra for the first variable (0)
        for t in self.args.val_steps_to_log:
            fname = self.save_spectrum(
                rad_truth=spectra_gt_tensor[t-1],
                rad_forecast=spectra_tensor[t-1],
                time=t,
                dir=wandb.run.dir
            )
            wandb.log({
                "spectrum": wandb.Image(fname)
            }, step=t)

        self.spectra_metrics["spectra"].clear()
        self.spectra_metrics["spectra_gt"].clear()
