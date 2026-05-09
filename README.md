# TiPAI-TSPO
->the script loads a said stable diffusion model into the stable diffusion piepline (use the last cell of stable-diff-pipeline as example) (specific model to be loaded in the pipeline will come from yaml file )
->at the last 30% timestep t from noise to image generation the latent at selected timestep is converted to image space and send to the auditor  (how many times to stop in last 30% will come from yaml file) 
     -> if the image is categorized as adversarial by the auditor 
     -> the image data i.e the 275 or some dim vector is sent to tournament_inference.py that uses the input features and the model weights inside TSPO weights to suggest candidate inputs to the inapinter. each call suggests one candidate so we need to make n calls (this value will be provided from a yaml file)
     -> these n candidate values mask along with the image and original prompt are given to the stable diffusion 1.5 inapinter that loads soft tensor values from inpainter-weights and produces n inpainted candidates . 
     -> this inpainted candidates + the original image is sent to the policy_inference where the policy selects one winner image of the tournament 
     ->This image is reinserted into the diffusion pipeline using null text inversion 
     ->diffusion denoising continues until next time step t 
->else if the auditor does not categorize the image as adversarial the diffusion denoising continues as is 


### How to make file configurations 
->TiPAI-TSPO-model
    ->attacks - store csv or txt file of attacks prompts 
    ->inpainter-weights - safe the LoRA softtensor weights for inpainter model 
    ->TSPO-weights - stores .pth weights file used by tournament_inference candidate generator model for inpainted    candidates
    

