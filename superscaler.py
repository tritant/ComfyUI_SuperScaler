import torch
import numpy as np
import comfy.utils
import nodes
import gc
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

# (Optionnel, mais recommandé pour le Post-FX si ton code l'utilise)
try:
    import cv2
except ImportError:
    print("Avertissement: OpenCV (cv2) n'est pas installé. Le module de sharpening pourrait ne pas fonctionner s'il en dépend.")

class SuperScaler:
    """
    Le node "Pipeline SuperScaler" tout-en-un.
    Combine un raffinage latent (Pass 1), deux passes d'upscale génératif tiled (Pass 2, Pass 3),
    et un post-traitement (Sharpening & Grain).
    """

    # --- Conversion Tenseur <-> NumPy (pour le Post-FX) ---
    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """Convertit un tenseur d'image ComfyUI (BCHW, float32) en NumPy (HWC, uint8)."""
        image_np = tensor.cpu().numpy().squeeze(0)  # (B, H, W, C) -> (H, W, C)
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        return image_np

    def np_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """Convertit un NumPy (HWC, uint8) en tenseur d'image ComfyUI (BCHW, float32)."""
        image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # (H, W, C) -> (B, H, W, C)
        return image_tensor

    # --- Définition du Node ---
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE",),
            },
            "optional": {
                # --- SECTION 1: LATENT REFINEMENT (PASS 1) ---
                "mask_in": ("MASK",),
                "enable_latent_pass": ("BOOLEAN", {"default": False}),
                "model_pass_1": ("MODEL",),
                "vae_pass_1": ("VAE",),
                "positive_pass_1": ("CONDITIONING",),
                "negative_pass_1": ("CONDITIONING",),
                "latent_upscale_by": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 4.0, "step": 0.1}),
                "latent_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "latent_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "latent_steps": ("INT", {"default": 6, "min": 1, "max": 1000}),
                "latent_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                                                
                # --- SECTION 2: TILED GENERATIVE UPSCALE (PASS 2) --- (MODIFIÉ: _2 ajouté)
                "enable_tiled_pass_2": ("BOOLEAN", {"default": True}),
                "model_pass_2": ("MODEL",),
                "vae_pass_2": ("VAE",),
                "positive_pass_2": ("CONDITIONING",),
                "negative_pass_2": ("CONDITIONING",),
                "tiled_upscale_by_2": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "tiled_denoise_2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_size_2": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap_2": ("INT", {"default": 64, "min": 32, "max": 512, "step": 32}),
                "tiled_sampler_name_2": (comfy.samplers.KSampler.SAMPLERS,),
                "tiled_scheduler_2": (comfy.samplers.KSampler.SCHEDULERS,),
                "tiled_steps_2": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "tiled_cfg_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                
                # --- SECTION 3: TILED GENERATIVE UPSCALE (PASS 3) --- (NOUVEAU)
                "enable_tiled_pass_3": ("BOOLEAN", {"default": False}),
                "model_pass_3": ("MODEL",),
                "vae_pass_3": ("VAE",),
                "positive_pass_3": ("CONDITIONING",),
                "negative_pass_3": ("CONDITIONING",),
                "tiled_upscale_by_3": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "tiled_denoise_3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_size_3": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap_3": ("INT", {"default": 64, "min": 32, "max": 512, "step": 32}),
                "tiled_sampler_name_3": (comfy.samplers.KSampler.SAMPLERS,),
                "tiled_scheduler_3": (comfy.samplers.KSampler.SCHEDULERS,),
                "tiled_steps_3": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "tiled_cfg_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                
                # --- SECTION 4: POST-PROCESSING FX ---
                "enable_sharpen": ("BOOLEAN", {"default": False}),
                "sharpen_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "sharpen_radius": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "enable_grain": ("BOOLEAN", {"default": False}),
                "grain_type": (["poisson", "gaussian", "perlin"], {"default": "poisson"}),
                "grain_intensity": ("FLOAT", {"default": 0.014, "min": 0.001, "max": 1.0, "step": 0.001}),
                "grain_size": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 16.0, "step": 0.1}),
                "saturation_mix": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaptive_grain": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask_blend_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "process"
    CATEGORY = "upscaling"
    
    # --- MODULE 1: RAFFINAGE LATENT ---
    def run_latent_refine(self, image, model, vae, positive, negative, 
                          upscale_by, denoise, sampler, scheduler, steps, cfg, seed):
        
        print(f"[SuperScaler] PASS 1: Raffinage Latent (x{upscale_by})")
        
        # 1. Encoder l'image
        latent = vae.encode(image)

        # 2. Upscaler le latent
        b, c, h, w = latent.shape
        new_h, new_w = int(h * upscale_by), int(w * upscale_by)
        latent_upscaled = torch.nn.functional.interpolate(latent, size=(new_h, new_w), mode="nearest")

        # 3. Appeler le KSampler
        refined_latent = nodes.common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent={"samples": latent_upscaled},
            denoise=denoise
        )[0] # [0] pour obtenir le latent de sortie

        # 4. Décoder
        refined_image = vae.decode(refined_latent["samples"])
        
        del latent, latent_upscaled, refined_latent
        gc.collect()
        torch.cuda.empty_cache()
        
        return refined_image

    # --- MODULE 2 & 3: UPSCALE TILED --- (MODIFIÉ: fonction générique)
    def _run_tiled_pass(self, pass_name, image, model, vae, positive, negative, 
                           upscale_by, denoise, tile_size, overlap, 
                           sampler, scheduler, steps, cfg, seed):
        
        print(f"[SuperScaler] {pass_name}: Upscale Tiled (x{upscale_by})")

        device = comfy.model_management.get_torch_device()
        image = image.to(device)

        # 1. Upscale "bête" (Pixel space)
        b, h, w, c = image.shape
        new_h, new_w = int(h * upscale_by), int(w * upscale_by)
        # Permuter pour NCHW pour l'interpolation
        image_nchw = image.permute(0, 3, 1, 2)
        image_blurry_large = torch.nn.functional.interpolate(
            image_nchw, size=(new_h, new_w), mode="bicubic", antialias=True
        )
        
        # Tenseur de sortie final, initialisé en noir
        output_image = torch.zeros_like(image_blurry_large)
        # Tenseur de "comptage" pour la moyenne des chevauchements
        overlap_count = torch.zeros_like(image_blurry_large)

        # 2. Créer le masque de fondu (feather mask)
        feather_mask = self._create_feather_mask(tile_size, tile_size, overlap, device=device)

        # 3. Boucle de Tiling
        stride = tile_size - overlap
        
        # --- Début des Logs ---
        y_steps = list(range(0, new_h, stride))
        x_steps = list(range(0, new_w, stride))
        total_tiles = len(y_steps) * len(x_steps)
        current_tile = 0
        print(f"[SuperScaler] {pass_name}: Démarrage de l'upscale, {total_tiles} tuiles à calculer...")
        # --- Fin des Logs ---
        
        for y in y_steps: # Utiliser les listes pré-calculées
            for x in x_steps: # Utiliser les listes pré-calculées
                
                # --- Ajout du compteur ---
                current_tile += 1
                print(f"[SuperScaler] {pass_name}: Inférence de la tuile {current_tile}/{total_tiles}...")
                # --- Fin du compteur ---

                # S'assurer que la tuile ne dépasse pas les bords
                y_end = min(y + tile_size, new_h)
                x_end = min(x + tile_size, new_w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                # Découper la tuile
                tile = image_blurry_large[:, :, y_start:y_end, x_start:x_end]

                # 4. KSampler sur la tuile (Encode -> KSample -> Decode)
                tile_latent = vae.encode(tile.permute(0, 2, 3, 1)) # NCHW -> NHWC
                
                tile_refined_latent = nodes.common_ksampler(
                    model=model,
                    seed=seed, # Utiliser la même seed pour la cohérence
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent={"samples": tile_latent},
                    denoise=denoise
                )[0]
                
                tile_refined = vae.decode(tile_refined_latent["samples"]) # Sortie en NHWC
                tile_refined = tile_refined.permute(0, 3, 1, 2).to(device) # NHWC -> NCHW ET move to GPU

                # 5. Appliquer le masque de fondu et ré-assembler
                current_mask = feather_mask
                # Gérer les tuiles de bord qui sont plus petites
                if tile_refined.shape[2] != tile_size or tile_refined.shape[3] != tile_size:
                    current_mask = self._create_feather_mask(tile_refined.shape[2], tile_refined.shape[3], overlap, device=device)
                    
                output_image[:, :, y_start:y_end, x_start:x_end] += tile_refined * current_mask
                overlap_count[:, :, y_start:y_end, x_start:x_end] += current_mask

                del tile, tile_latent, tile_refined_latent, tile_refined
                
        # Normaliser les zones de chevauchement
        final_image = (output_image / overlap_count)
        # Remplacer tous les NaN (résultats de 0/0) par 0.0 (noir)
        final_image = torch.nan_to_num(final_image, nan=0.0)
        
        del image_blurry_large, output_image, overlap_count, feather_mask
        gc.collect()
        torch.cuda.empty_cache()
        
        # Re-permuter en NHWC (format IMAGE Comfy)
        return final_image.permute(0, 2, 3, 1).cpu()

    def _create_feather_mask(self, h, w, overlap, device="cpu"):
        """Crée un masque de fondu (linear blend) pour le tiling."""
        overlap_x = min(overlap, w // 2)
        overlap_y = min(overlap, h // 2)

        # Créer les rampes linéaires
        linear_x = torch.ones(w, device=device)
        if overlap_x > 0:
            linear_x[:overlap_x] = torch.linspace(0, 1, overlap_x, device=device)
            linear_x[-overlap_x:] = torch.linspace(1, 0, overlap_x, device=device)

        linear_y = torch.ones(h, device=device)
        if overlap_y > 0:
            linear_y[:overlap_y] = torch.linspace(0, 1, overlap_y, device=device)
            linear_y[-overlap_y:] = torch.linspace(1, 0, overlap_y, device=device)

        # Combiner en 2D
        mask_2d = torch.outer(linear_y, linear_x)
        return mask_2d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)


    # --- MODULE 4: POST-FX ---
    def run_freq_split_sharpen(self, image_tensor, amount, radius):
        strength = amount
        print(f"[SuperScaler] PASS 4.1: Sharpening (Amount: {amount}, Radius: {radius})")
        
        if strength == 0:
            return (image_tensor,) # Retourne l'original si force = 0

        # --- TON CODE INSÉRÉ ICI ---
        # Note: 'image_tensor' est déjà un float 0..1 (BHWC)
        image_np = image_tensor.clone().numpy()[0] # BHWC -> HWC
        low_freq = gaussian_filter(image_np, sigma=(radius, radius, 0))
        high_freq = image_np - low_freq
        processed_high_freq = high_freq * strength
        final_image_np = low_freq + processed_high_freq
        final_image_np = np.clip(final_image_np, 0, 1)
        final_image_tensor = torch.from_numpy(final_image_np).unsqueeze(0) # HWC -> BHWC
        # --- FIN DE TON CODE ---

        return final_image_tensor

    def run_add_grain(self, images, grain_type, grain_intensity, grain_size, saturation_mix, adaptive_grain):
        print(f"[SuperScaler] PASS 4.2: Grain (Type: {grain_type}, Intensity: {grain_intensity})")
        
        device = comfy.model_management.get_torch_device()
        images = images.to(device) # Doit être un Tenseur (B, H, W, C)

        if grain_intensity == 0.0:
            return images

        if grain_type == "gaussian":
            grain = self._generate_gaussian(images, grain_size)
        elif grain_type == "poisson":
            grain = self._generate_poisson(images, grain_size)
        elif grain_type == "perlin":
            grain = self._generate_perlin(images, grain_size)
        else:
            # Fallback au cas où
            print(f"[SuperScaler] Type de grain inconnu '{grain_type}', utilisation de 'poisson'")
            grain = self._generate_poisson(images, grain_size)

        # Logique de Saturation Mix
        gray = grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
        grain = saturation_mix * grain + (1.0 - saturation_mix) * gray

        # Logique Adaptive Grain
        if adaptive_grain > 0.0:
            luma = images.mean(dim=3, keepdim=True)
            gain = (1.0 - luma).pow(2.0) * 2.5
            grain = grain * (1.0 + adaptive_grain * gain)
        
        # Appliquer le grain
        output = images + grain * grain_intensity
        output = output.clamp(0.0, 1.0)

        # Retourne le tenseur (pas un tuple)
        return output.to(comfy.model_management.intermediate_device())
        
    def _generate_gaussian(self, images, size):
        B, H, W, C = images.shape
        size = int(size) 
        if size <= 1:
            noise = torch.randn_like(images)
        else:
            small_H, small_W = H // size, W // size
            noise = torch.randn(B, small_H, small_W, C, device=images.device)
            noise = noise.permute(0, 3, 1, 2)
            noise = F.interpolate(noise, size=(H, W), mode="nearest")
            noise = noise.permute(0, 2, 3, 1)
        
        noise[:, :, :, 0] *= 2.0
        noise[:, :, :, 2] *= 3.0
        return noise

    def _generate_poisson(self, images, size):
        B, H, W, C = images.shape
        size = int(size) 
        if size <= 1:
            target_images = images
        else:
            small_H, small_W = H // size, W // size
            target_images = F.interpolate(images.permute(0, 3, 1, 2), size=(small_H, small_W), mode="bilinear", align_corners=False)
            target_images = target_images.permute(0, 2, 3, 1)

        scaled = torch.clamp(target_images * 255.0, 0, 255).round()
        noise = torch.poisson(scaled) - scaled
        grain = (noise / 255.0) * 16.0

        if size > 1:
            grain = grain.permute(0, 3, 1, 2)
            grain = F.interpolate(grain, size=(H, W), mode="nearest")
            grain = grain.permute(0, 2, 3, 1)

        grain[:, :, :, 0] *= 2.0
        grain[:, :, :, 2] *= 3.0
        return grain

    def _generate_perlin(self, images, size):
        B, H, W, C = images.shape
        size = int(size)
        scale = max(4, 32 // size)
        
        perlin = self._make_fractal_noise(B, H, W, scale, images.device)
        perlin = (perlin - 0.5) * 2.0
        
        perlin = perlin.unsqueeze(3).repeat(1, 1, 1, 3)
        return perlin

    def _make_fractal_noise(self, B, H, W, scale, device, octaves=4, persistence=0.5, lacunarity=2.0):
        total_noise = torch.zeros(B, H, W, device=device)
        frequency = 1.0
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            current_scale = max(2, int(scale * frequency))
            coarse_noise = torch.rand(B, current_scale, current_scale, 1, device=device)
            upscaled_noise = F.interpolate(coarse_noise.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            total_noise += upscaled_noise * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            total_noise /= max_amplitude
            
        return total_noise


    # --- FONCTION PRINCIPALE ---
    def process(self, image_in, **kwargs):
    
        # --- NOUVEAU : Sauvegarde de l'image originale ---
        original_clean = image_in.clone()
        
        # Récupérer les nouveaux arguments
        mask_in = kwargs.get("mask_in", None)
        mask_blend_weight = kwargs.get("mask_blend_weight", 1.0)
        
        global_seed = kwargs.get("seed", 0)
        # Récupérer tous les arguments
        enable_latent_pass = kwargs.get("enable_latent_pass", False)
        model_pass_1 = kwargs.get("model_pass_1", None)
        vae_pass_1 = kwargs.get("vae_pass_1", None)
        positive_pass_1 = kwargs.get("positive_pass_1", None)
        negative_pass_1 = kwargs.get("negative_pass_1", None)
        
        # (MODIFIÉ: _2 ajouté)
        enable_tiled_pass_2 = kwargs.get("enable_tiled_pass_2", False)
        model_pass_2 = kwargs.get("model_pass_2", None)
        vae_pass_2 = kwargs.get("vae_pass_2", None)
        positive_pass_2 = kwargs.get("positive_pass_2", None)
        negative_pass_2 = kwargs.get("negative_pass_2", None)

        # (NOUVEAU)
        enable_tiled_pass_3 = kwargs.get("enable_tiled_pass_3", False)
        model_pass_3 = kwargs.get("model_pass_3", None)
        vae_pass_3 = kwargs.get("vae_pass_3", None)
        positive_pass_3 = kwargs.get("positive_pass_3", None)
        negative_pass_3 = kwargs.get("negative_pass_3", None)

        enable_sharpen = kwargs.get("enable_sharpen", False)
        enable_grain = kwargs.get("enable_grain", False)
        
        # Démarrer le pipeline
        current_image = image_in.clone()

        # --- Exécuter PASS 1 ---
        if (enable_latent_pass and 
            model_pass_1 is not None and 
            vae_pass_1 is not None and 
            positive_pass_1 is not None and 
            negative_pass_1 is not None):
            
            current_image = self.run_latent_refine(
                image=current_image,
                model=model_pass_1,
                vae=vae_pass_1,
                positive=positive_pass_1,
                negative=negative_pass_1,
                upscale_by=kwargs.get("latent_upscale_by", 1.5),
                denoise=kwargs.get("latent_denoise", 0.5),
                sampler=kwargs.get("latent_sampler_name", "dpmpp_2m"),
                scheduler=kwargs.get("latent_scheduler", "karras"),
                steps=kwargs.get("latent_steps", 20),
                cfg=kwargs.get("latent_cfg", 7.0),
                seed=global_seed
            )
        else:
            print("[SuperScaler] PASS 1: Sauté (désactivé ou inputs manquants)")

        # --- Exécuter PASS 2 --- (MODIFIÉ: _2 ajouté)
        if (enable_tiled_pass_2 and 
            model_pass_2 is not None and 
            vae_pass_2 is not None and 
            positive_pass_2 is not None and 
            negative_pass_2 is not None):
            
            current_image = self._run_tiled_pass(
                pass_name="PASS 2",
                image=current_image,
                model=model_pass_2,
                vae=vae_pass_2,
                positive=positive_pass_2,
                negative=negative_pass_2,
                upscale_by=kwargs.get("tiled_upscale_by_2", 2.0),
                denoise=kwargs.get("tiled_denoise_2", 0.35),
                tile_size=kwargs.get("tile_size_2", 768),
                overlap=kwargs.get("tile_overlap_2", 128),
                sampler=kwargs.get("tiled_sampler_name_2", "dpmpp_2m"),
                scheduler=kwargs.get("tiled_scheduler_2", "karras"),
                steps=kwargs.get("tiled_steps_2", 15),
                cfg=kwargs.get("tiled_cfg_2", 7.0),
                seed=global_seed
            )
        else:
            print("[SuperScaler] PASS 2: Sauté (désactivé ou inputs manquants)")
            
        # --- Exécuter PASS 3 --- (NOUVEAU)
        if (enable_tiled_pass_3 and 
            model_pass_3 is not None and 
            vae_pass_3 is not None and 
            positive_pass_3 is not None and 
            negative_pass_3 is not None):
            
            current_image = self._run_tiled_pass(
                pass_name="PASS 3",
                image=current_image,
                model=model_pass_3,
                vae=vae_pass_3,
                positive=positive_pass_3,
                negative=negative_pass_3,
                upscale_by=kwargs.get("tiled_upscale_by_3", 2.0),
                denoise=kwargs.get("tiled_denoise_3", 0.15),
                tile_size=kwargs.get("tile_size_3", 512),
                overlap=kwargs.get("tile_overlap_3", 64),
                sampler=kwargs.get("tiled_sampler_name_3", "dpmpp_2m"),
                scheduler=kwargs.get("tiled_scheduler_3", "karras"),
                steps=kwargs.get("tiled_steps_3", 12),
                cfg=kwargs.get("tiled_cfg_3", 7.0),
                seed=global_seed
            )
        else:
            print("[SuperScaler] PASS 3: Sauté (désactivé ou inputs manquants)")

        # --- Exécuter PASS 4 (POST-FX) ---
        if enable_sharpen:
            current_image = self.run_freq_split_sharpen(
                image_tensor=current_image,
                amount=kwargs.get("sharpen_amount", 0.5),
                radius=kwargs.get("sharpen_radius", 2)
            )
            
        if enable_grain:
            current_image = self.run_add_grain(
                images=current_image,
                grain_type=kwargs.get("grain_type", "poisson"),
                grain_intensity=kwargs.get("grain_intensity", 0.022),
                grain_size=kwargs.get("grain_size", 1.5),
                saturation_mix=kwargs.get("saturation_mix", 0.22),
                adaptive_grain=kwargs.get("adaptive_grain", 0.30)
            )
            
        # --- NOUVEAU : PASS 5 (FINAL MASKED BLEND) ---
        # current_image est notre image finale "traitée" (sur CPU)
        # On vérifie si un masque est fourni ET si le poids est > 0
        if mask_in is not None and mask_blend_weight > 0.0:
            print(f"[SuperScaler] PASS 5: Mélange final avec masque (Poids: {mask_blend_weight})")
            
            device = comfy.model_management.get_torch_device()
            
            # 1. Préparer l'image finale traitée (sur GPU)
            final_processed = current_image.to(device)
            B, H, W, C = final_processed.shape

            # 2. Préparer l'image originale "propre" (upscalée simplement)
            original_clean_nchw = original_clean.to(device).permute(0, 3, 1, 2)
            original_clean_upscaled_nchw = F.interpolate(
                original_clean_nchw, 
                size=(H, W), 
                mode="bicubic", 
                antialias=True
            )
            original_clean_upscaled = original_clean_upscaled_nchw.permute(0, 2, 3, 1)

            # 3. Préparer le masque (redimensionné à la taille finale)
            # Le masque est (B, H, W), on doit le passer en (B, 1, H, W) pour interpolate
            mask_nchw = mask_in.to(device).reshape(B, 1, mask_in.shape[1], mask_in.shape[2])
            mask_resized_nchw = F.interpolate(
                mask_nchw, 
                size=(H, W), 
                mode="bilinear", 
                align_corners=False
            )
            # Re-permuter en (B, H, W, 1) pour le blending
            mask_final = mask_resized_nchw.permute(0, 2, 3, 1)
            
            # 4. Logique de mélange
            # Définir ce qu'est la "zone protégée" (le ciel)
            # Le poids contrôle à quel point la zone protégée est "propre"
            protected_image = (original_clean_upscaled * mask_blend_weight) + \
                              (final_processed * (1.0 - mask_blend_weight))
                              
            # Logique INVERSÉE comme demandé :
            # Masque NOIR (0.0) -> protected_image
            # Masque BLANC (1.0) -> final_processed
            current_image = (final_processed * (1.0 - mask_final)) + (protected_image * mask_final)
            
            # Nettoyage
            del final_processed, original_clean_nchw, original_clean_upscaled, mask_nchw, mask_resized_nchw, mask_final
            gc.collect()
            torch.cuda.empty_cache()
            
            # Retourner l'image finale depuis le GPU vers le CPU
            return (current_image.cpu(),)    
            
        return (current_image,)

# --- Enregistrement du Node ---
NODE_CLASS_MAPPINGS = {
    "SuperScaler_Pipeline": SuperScaler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperScaler_Pipeline": "Pipeline SuperScaler"
}