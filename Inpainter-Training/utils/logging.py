import wandb

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

    init_kwargs = {
        "project": project,
        "config": config,
        "mode": mode,
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

    return wandb.init(**init_kwargs)

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