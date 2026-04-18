import os

import wandb


def _has_netrc_wandb_login(netrc_path="/root/.netrc"):
    """Return True if the netrc file has credentials for api.wandb.ai."""
    if not os.path.exists(netrc_path):
        return False
    try:
        with open(netrc_path, "r", encoding="utf-8") as f:
            text = f.read()
        return "machine api.wandb.ai" in text and "login" in text
    except OSError:
        return False

def init_wandb(project, config):
    """Initialize Weights & Biases logging.
    
    Sets up wandb project and logs the configuration for the training run.
    
    Args:
        project (str): The wandb project name.
        config (dict): Configuration dictionary from YAML config file.
    
    Returns:
        wandb.run: The wandb run object for the current session.
    """
    output_cfg = config.get("output", {}) if isinstance(config, dict) else {}
    enabled = output_cfg.get("use_wandb", True)
    mode = output_cfg.get("wandb_mode")

    if mode is None:
        mode = "online" if enabled else "disabled"

    # Skip online init when no credentials are present to avoid long startup waits.
    if mode == "online" and not (os.getenv("WANDB_API_KEY") or _has_netrc_wandb_login()):
        print("[wandb] No API key found (WANDB_API_KEY or /root/.netrc). Using disabled mode.")
        mode = "disabled"

    init_kwargs = {
        "project": project,
        "config": config,
        "mode": mode,
        "settings": wandb.Settings(init_timeout=20),
    }

    entity = output_cfg.get("wandb_entity")
    run_name = output_cfg.get("wandb_run_name")
    tags = output_cfg.get("wandb_tags")

    if entity:
        init_kwargs["entity"] = entity
    if run_name:
        init_kwargs["name"] = run_name
    if tags:
        init_kwargs["tags"] = tags

    try:
        return wandb.init(**init_kwargs)
    except Exception as err:
        # Keep training running even if wandb auth/network init fails.
        print(f"[wandb] Initialization failed ({err}). Falling back to disabled mode.")
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "disabled"
        fallback_kwargs.pop("entity", None)
        fallback_kwargs["settings"] = wandb.Settings(init_timeout=5)
        return wandb.init(**fallback_kwargs)

def log_metrics(step, metrics):
    """Log training metrics to Weights & Biases.
    
    Args:
        step (int): The current training step number.
        metrics (dict): Dictionary of metric names and values to log.
    """
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)

def finish_wandb():
    """Finish the Weights & Biases logging session.
    
    Closes the wandb run and uploads final logs.
    """
    if wandb.run is None:
        return
    wandb.finish()