# SuperScaler Pipeline for ComfyUI

An all-in-one, multi-pass generative upscaling and post-processing node designed to simplify complex workflows and add a professional finish to your images.

This node replaces a complex chain of 10+ nodes, combining a latent refine pass, two generative tiled upscale passes, and a final Post-FX (Sharpen & Grain) stage into a single, convenient, and collapsible unit.

-----
<img width="2556" height="1217" alt="Capture d&#39;écran 2025-11-02 072618" src="https://github.com/user-attachments/assets/09dc363d-e478-4435-a4b4-3730b9127ca3" />

---

### 🌟 New Feature: Masked Final Blending

This node now includes an optional `mask_in` input and a `mask_blend_weight` slider under **"Final Settings"**.

This powerful feature allows you to protect specific areas of your image (like skies or smooth surfaces) from the entire generative and post-processing pipeline, preventing unwanted noise or artifacts in those regions.

The `mask_blend_weight` slider controls the "cleanliness" of the protected areas.
* At **1.0** (default), the protected area is 100% replaced by the clean upscale.
* At **0.5**, the protected area is a 50/50 mix of the clean and processed versions.
* At **0.0**, the mask has no effect.

## 🌟 Features

  * **All-in-One Pipeline:** A single node to take your image from 1K to 4K+ with generative details, sharpening, and realistic film grain.
  * **Dynamic & Collapsible UI:** Tidy up your workflow\! Each major pass can be toggled on or off, hiding its settings to save space.
  * **Multi-Pass Upscaling:**
      * **Pass 1: Latent Refine:** A subtle `img2img`-style pass to refine the base image before the main upscale.
      * **Pass 2 & 3: Tiled Generative Upscale:** Two independent generative (img2img) passes that use tiling to perform large upscales (e.g., 2x + 2x) on low-VRAM GPUs.
  * **Professional Post-FX Stage:**
      * **Frequency-Split Sharpening:** A high-quality sharpening method (`scipy.ndimage.gaussian_filter`) that sharpens details without introducing harsh halos.
      * **Realistic Film Grain:** Adds `poisson`, `gaussian`, or `perlin` grain to break up digital smoothness, add texture, and make images feel more realistic and less "plastic."
  * **Global Seed Control:** A single, globally accessible seed controls all three generative passes, ensuring consistent and reproducible results.

-----

## ⚙️ Node Parameters (The Pipeline)

The node is organized into sections that follow the image processing pipeline.

### 1\. Latent Pass (Pass 1)

  * **Purpose:** A gentle, non-tiled `img2img` pass.
  * **Use Case:** Ideal for slightly modifying or refining the base image *before* the main upscale. Uses a low `denoise` (e.g., 0.4) and a small `upscale_by` (e.g., 1.1x).

### 2\. Tiled Pass 2 (Pass 2)

  * **Purpose:** The primary generative upscale pass.
  * **Use Case:** This is your main 2x or 3x upscale. It uses tiling, so you can use large `tile_size` (like 768) with low VRAM. A moderate `denoise` (e.g., 0.35) will add generative details.

### 3\. Tiled Pass 3 (Pass 3)

  * **Purpose:** A second, optional generative upscale pass.
  * **Use Case:** Use this for a 4x \> 8x upscale. It takes the output of Pass 2 and tiles it again. Typically, this pass uses a much lower `denoise` (e.g., 0.15) to refine details without changing the image.

### 4\. Post-FX (Pass 4)

This stage is applied in pixel-space at the very end.

  * **Enable Sharpen:**
      * **Amount:** Strength of the sharpening effect.
      * **Radius:** How wide the detail-detection radius is.
  * **Enable Grain:**
      * **Grain Intensity:** The overall strength/visibility of the grain.
      * **Grain Type:** `poisson` (default, very organic), `gaussian`, or `perlin`.
      * **Grain Size:** The scale of the grain particles.
      * **Saturation Mix:** How colorful the grain is (1.0 = full color, 0.0 = monochrome).
      * **Adaptive Grain:** Ties the grain strength to the image's brightness (more grain in shadows).

### 5\. Global Seed

  * **Purpose:** Controls all three generative passes (Pass 1, 2, and 3) simultaneously for consistent results.
  * **Features:** Includes the standard ComfyUI `fixed`, `increment`, `randomize` controls.
