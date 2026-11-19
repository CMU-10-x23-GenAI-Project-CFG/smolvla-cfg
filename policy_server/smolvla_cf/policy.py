import json
import pathlib

import numpy as np
import torch
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smolvla_cf import guidance

NEGATIVES = json.loads((pathlib.Path(__file__).parent / "negatives.json").read_text())


class SmolVLA:
    def __init__(self, scale=1.0, null="none", schedule=None):
        torch.manual_seed(7)
        self.scale = scale
        self.null = null
        # Optional per-step strength schedule (guidance.Schedule). When set,
        # inference guides the velocity at every Euler step via
        # guidance.guided_sample_actions instead of mixing the final chunk.
        # When None, behaviour is unchanged (final-chunk CAG mix below).
        self.schedule = schedule
        self.policy = SmolVLAPolicy.from_pretrained("k1000dai/smolvla_libero_finetune").eval()
        self.device = self.policy.config.device

    def to_batch(self, obs):
        return {
            OBS_STATE: torch.tensor(np.array(obs["observation/state"]), dtype=torch.float32, device=self.device)[None],
            "observation.images.image": self.to_image(obs["observation/image"]),
            "observation.images.wrist_image": self.to_image(obs["observation/wrist_image"]),
            "task": str(obs["prompt"]),
        }

    def to_image(self, image):
        return torch.from_numpy(np.array(image)).to(self.device).permute(2, 0, 1)[None].float() / 255.0

    @torch.no_grad()
    def infer(self, obs):
        batch = self.policy._prepare_batch(self.to_batch(obs))
        images, image_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        tokens, token_masks = self.policy.prepare_language(batch)
        config = self.policy.config
        noise = torch.randn(1, config.chunk_size, config.max_action_dim, device=self.device)

        def sample(tokens, masks):
            return self.policy.model.sample_actions(images, image_masks, tokens, masks, state, noise=noise.clone())

        def null_language():
            if self.null == "negative":
                return self.policy.prepare_language({**batch, "task": NEGATIVES[batch["task"]]})
            return torch.zeros_like(tokens), torch.zeros_like(token_masks)

        if self.schedule is not None and self.null != "none":
            # Per-step guidance: mix the velocity at every Euler step
            # (guidance Eq. 1) rather than only the final action chunk.
            null_tokens, null_masks = null_language()
            actions = guidance.guided_sample_actions(
                self.policy.model, images, image_masks,
                tokens, token_masks, null_tokens, null_masks,
                state, noise.clone(), self.schedule,
            )
        else:
            actions = sample(tokens, token_masks)
            if self.null != "none":
                null_actions = sample(*null_language())
                actions = null_actions + self.scale * (actions - null_actions)
        actions = actions[:, :, : config.action_feature.shape[0]]
        actions = self.policy.unnormalize_outputs({ACTION: actions})[ACTION]
        return {"actions": actions[0].cpu().numpy()}
