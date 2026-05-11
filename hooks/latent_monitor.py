import torch
from transformer_lens import HookedTransformer

class LatentMonitor:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model = HookedTransformer.from_pretrained(model_name, device="cuda")
        # In a real run, we'd load your pre-trained SAE here
        self.sycophancy_vector = torch.randn(self.model.cfg.d_model).cuda() 

    def get_sycophancy_score(self, activations):
        """Calculates cosine similarity to the sycophancy/authority direction."""
        # Simple linear probe logic for the MVP
        similarity = torch.cosine_similarity(activations, self.sycophancy_vector, dim=-1)
        return similarity.mean().item()

    def run_with_monitoring(self, prompt):
        """Runs the model and captures residual stream telemetry per token."""
        scores = []
        
        def hook_fn(value, hook):
            # Capture the residual stream at a mid-layer (e.g., Layer 12)
            score = self.get_sycophancy_score(value)
            scores.append(score)

        # Hook into the mid-point of the residual stream
        self.model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{self.model.cfg.n_layers//2}.hook_resid_post", hook_fn)]
        )
        return scores