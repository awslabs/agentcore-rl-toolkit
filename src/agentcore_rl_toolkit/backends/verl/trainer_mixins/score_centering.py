"""Train with score function centering when ``score_centering.num_auxiliary_token_samples > 0``."""

from __future__ import annotations

from functools import partial

from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass

from .. import score_centering
from .base import TrainerMixinBase


def _validate(config, parameter_sync_step: int) -> None:
    actor, rollout, model = (
        config.actor_rollout_ref.actor,
        config.actor_rollout_ref.rollout,
        config.actor_rollout_ref.model,
    )
    mtp = rollout.get("mtp")
    rollout_is = (config.algorithm.get("rollout_correction") or {}).get("rollout_is")
    one_update = (
        actor.ppo_epochs == 1 and actor.ppo_mini_batch_size * parameter_sync_step == config.data.train_batch_size
    )
    checks = {
        f"export {score_centering.SAMPLES_ENV}=1": score_centering.samples_enabled(),
        "rollout.name=vllm": rollout.name == "vllm",
        "rollout.logprobs_mode=processed_logprobs": rollout.get("logprobs_mode") == "processed_logprobs",
        "no speculative decoding": not (mtp and mtp.get("enable") and mtp.get("enable_rollout")),
        "actor.strategy fsdp or fsdp2": actor.strategy in ("fsdp", "fsdp2"),
        "actor.policy_loss.loss_mode=vanilla": actor.policy_loss.get("loss_mode", "vanilla") == "vanilla",
        "no distillation": not is_distillation_enabled(config.get("distillation")),
        "algorithm.rollout_correction.rollout_is=null": rollout_is is None,
        "actor.ppo_epochs=1 and ppo_mini_batch_size * parameter_sync_step = train_batch_size": one_update,
        "model.use_remove_padding=true": model.get("use_remove_padding", False),
        "model.use_fused_kernels=false": not model.get("use_fused_kernels", False),
    }
    if failed := [name for name, ok in checks.items() if not ok]:
        raise ValueError("score_centering requires: " + "; ".join(failed))


class ScoreCenteringMixin(TrainerMixinBase):
    """Install ``score_centering_loss`` on the actor and feed it the trainer's step."""

    def init(self):
        self._num_auxiliary_token_samples, self._kl_beta = score_centering.sfc_settings(self.config)
        if self._num_auxiliary_token_samples:
            _validate(self.config, self.parameter_sync_step)
            score_centering.install_padding_patch()
        super().init()

    def on_init_end(self):
        super().on_init_end()
        if not self._num_auxiliary_token_samples:
            return
        actor_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        self.actor_rollout_wg.set_loss_fn(partial(score_centering.score_centering_loss, config=actor_config))
        update_actor = self.actor_rollout_wg.update_actor

        def update_actor_with_logits(batch):
            batch.extra_info.update(
                {
                    "distillation_use_topk": True,  # verl's flag for calling the loss on logits
                    score_centering.KL_BETA_KEY: self._kl_beta,
                    # Weights version g - 1 are live during global step g.
                    score_centering.TRAINER_STEP_KEY: (self.global_steps - 1) * self.parameter_sync_step
                    + self.local_trigger_step,
                    score_centering.SYNC_STEP_KEY: self.parameter_sync_step,
                }
            )
            return update_actor(batch)

        self.actor_rollout_wg.update_actor = update_actor_with_logits
