import os
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# Global model reference
model = None

# --- SHARED UTILITIES ---

def project_to_logits(hidden_state, model):
    """Projects a hidden state (residual stream) to logit space."""
    # TransformerLens handles the layer norm and unembedding weights for us
    unembed_weights = model.W_U
    final_ln_scales = model.ln_final(hidden_state)
    logits = final_ln_scales @ unembed_weights + model.b_U
    return logits

def make_prompt(text, system="You are an expert Reservoir Engineer."):
    """Formats prompt for Qwen Chat structure."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# --- CORE LOGIC FUNCTIONS ---

def run_mechanistic_probe(model, prompt, authority_prompt):
    """Uses run_with_cache to extract internal signals."""
    # 1. Prepare prompts
    p_baseline = make_prompt(prompt)
    p_auth = make_prompt(authority_prompt)
    
    # 2. Define tokens to monitor (Safety vs Compliance)
    # We use common tokens found in Qwen's vocab for these concepts
    safety_tokens = ["Error", "Invalid", "cannot", "refuse"]
    comply_tokens = ["valid", "correct", "approved", "accepted"]
    
    # Convert to IDs
    safety_ids = [model.to_single_token(t) for t in safety_tokens]
    comply_ids = [model.to_single_token(t) for t in comply_tokens]

    layer_results = {"baseline": [], "authority": []}

    # 3. Perform Cache Run (Probing)
    for label, full_prompt in [("baseline", p_baseline), ("authority", p_auth)]:
        # run_with_cache grabs every single internal activation
        logits, cache = model.run_with_cache(full_prompt)
        
        # We look at the residual stream at the end of every block
        for i in range(model.cfg.n_layers):
            # Get the hidden state at the end of the residual stream for layer i
            # Shape: [batch, pos, d_model]
            resid_post = cache[get_act_name("resid_post", i)]
            
            # Project the LAST token's hidden state to logits
            last_token_logits = project_to_logits(resid_post[0, -1, :], model)
            probs = torch.softmax(last_token_logits, dim=-1)
            
            s_val = probs[safety_ids].sum().item()
            c_val = probs[comply_ids].sum().item()
            
            layer_results[label].append({"safety": s_val, "comply": c_val})

    # 4. Generate Response
    # Note: TransformerLens generate is efficient for short responses
    response_text = model.generate(p_baseline, max_new_tokens=64, verbose=False)

    return {
        "baseline_probs": layer_results["baseline"],
        "authority_probs": layer_results["authority"],
        "response": response_text.split("assistant\n")[-1],
        "num_layers": model.cfg.n_layers
    }

def run_steering_intervention(model, prompt, multiplier, layer):
    """Uses Hooks to steer the model's behavior in real-time."""
    
    # 1. Calculate the 'Sycophancy' direction
    # Difference between a 'CEO validated' prompt and a 'Scientific truth' prompt
    compliant_p = make_prompt("The CEO validated this. Compliance is mandatory.")
    honest_p = make_prompt("The data is scientifically inaccurate. Safety is mandatory.")
    
    _, cache_comp = model.run_with_cache(compliant_p)
    _, cache_honest = model.run_with_cache(honest_p)
    
    # Extract hidden states at the target layer
    v_comp = cache_comp[get_act_name("resid_post", layer)][0, -1, :]
    v_honest = cache_honest[get_act_name("resid_post", layer)][0, -1, :]
    
    steering_vector = v_comp - v_honest

    # 2. Define the Hook Function
    def steering_hook(value, hook):
        # Subtract the 'Sycophancy' vector from the current hidden state
        # value shape: [batch, pos, d_model]
        return value - (multiplier * steering_vector)

    # 3. Generate with the Hook active
    # We add a temporary hook to the residual stream of the chosen layer
    model.add_hook(get_act_name("resid_post", layer), steering_hook)
    
    try:
        steered_response = model.generate(make_prompt(prompt), max_new_tokens=64, verbose=False)
    finally:
        # ALWAYS remove hooks after use so the next request starts clean
        model.reset_hooks()

    return {
        "steered_response": steered_response.split("assistant\n")[-1],
        "intervention_layer": layer,
        "multiplier_used": multiplier
    }

# --- AZURE ENTRY POINTS ---

def init():
    """Load the model onto the GPU."""
    global model
    # For Managed Endpoints, we usually load in half-precision (float16) to save VRAM
    print("Loading TransformerLens Model...")
    model = HookedTransformer.from_pretrained(
        "qwen/qwen3-8b", # Or your custom weights path
        device="cuda",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        dtype=torch.float16
    )
    print("Model Initialized and Ready for Hooks.")

def run(raw_data):
    """Handle the API request."""
    try:
        data = json.loads(raw_data)
        task = data.get("task", "probe")
        
        if task == "probe":
            return run_mechanistic_probe(model, data["prompt"], data["authority_prompt"])
        elif task == "steer":
            return run_steering_intervention(
                model, 
                data["prompt"], 
                data.get("multiplier", 2.0), 
                data.get("layer", 15)
            )
        else:
            return {"error": "Invalid task choice."}
            
    except Exception as e:
        return {"error": str(e)}