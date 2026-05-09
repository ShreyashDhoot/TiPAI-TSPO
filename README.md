# TiPAI‑TSPO

- The script loads a specified Stable Diffusion model into the Stable Diffusion pipeline (use the last cell of `stable-diff-pipeline` as an example).  
  The specific model to be loaded is read from a YAML file.

- During the last 30% timesteps (from noise to image generation):  
  - At selected timesteps `t`, the latent is converted to image space and sent to the **auditor**.  
    (How many times to stop in the last 30% is configured via the YAML file.)
  - If the image is categorized as **adversarial** by the auditor:
    - The image data (e.g., a 275‑dimensional or similar‑size vector) is sent to `tournament_inference.py`.
    - `tournament_inference.py` uses:
      - the input features, and  
      - the model weights inside **TSPO‑weights**,  
      to suggest candidate inputs to the inpainter.
    - Each call of `tournament_inference.py` produces one candidate, so `n` calls are made  
      (the value `n` is provided from a YAML file).
    - These `n` candidate masks, along with the original image and prompt, are given to the **Stable Diffusion 1.5 inpainter**.
    - The inpainter loads soft‑tensor LoRA weights from `inpainter-weights` and produces `n` inpainted candidates.
    - The inpainted candidates plus the original image are sent to `policy_inference`, which:
      - runs a tournament and  
      - selects one winner image.
    - The winner image is re‑inserted into the diffusion pipeline using **null text inversion**.
    - Diffusion denoising continues until the next timestep `t`.
  - Else, if the auditor does **not** categorize the image as adversarial:
    - Diffusion denoising continues as normal, without any rewrite.

### How to make file configurations

- `TiPAI‑TSPO‑model/`
  - `attacks/`  
    Store CSV or TXT files of attack prompts.
  - `inpainter-weights/`  
    Save the LoRA soft‑tensor weights for the inpainter model.
  - `TSPO-weights/`  
    Store the `.pth` weights file used by `tournament_inference.py` (the candidate‑generator model) for inpainted candidates.

