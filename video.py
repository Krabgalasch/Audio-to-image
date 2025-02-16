import warnings
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import cv2
import numpy as np
import textwrap

from sentiment import phrases_with_emotion


# Combine frames into a video
def frames_to_video(frames, output_video="morphing_video.mp4", fps=30):
    """
    Combine frames into a video.
    """
    if not frames:
        print("No frames to create a video.")
        return

    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        # OpenCV expects BGR, so convert from RGB.
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Video saved as {output_video}")


# Overlay text on an image/frame
def add_text_to_frame(
    image: np.ndarray, text: str, font_scale: float = 1.0, font_thickness: int = 2
) -> np.ndarray:
    """
    Overlay the given text on the provided image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10

    # Wrap the text to a maximum width
    wrapped_text = textwrap.wrap(text, width=40)
    text_sizes = [
        cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        for line in wrapped_text
    ]
    if text_sizes:
        max_line_width = max(size[0] for size in text_sizes)
        total_text_height = (
            sum(size[1] for size in text_sizes) + (len(wrapped_text) - 1) * 5
        )
    else:
        max_line_width = 0
        total_text_height = 0

    # Create a semi-transparent rectangle as background
    overlay = image.copy()
    rect_x1, rect_y1 = margin - 5, margin - 5
    rect_x2, rect_y2 = margin + max_line_width + 5, margin + total_text_height + 5
    cv2.rectangle(
        overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), thickness=-1
    )
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw each line of text
    y = margin + text_sizes[0][1] if text_sizes else margin
    for i, line in enumerate(wrapped_text):
        cv2.putText(
            image,
            line,
            (margin, y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        y += text_sizes[i][1] + 5
    return image


# Generate final image and capture real intermediate diffusion frames


def generate_final_and_diffusion_frames(
    pipe, prompt, num_inference_steps=50, guidance_scale=7.5
):
    """
    Run the pipeline once while capturing all intermediate diffusion steps via a callback.
    Returns:
        (final_image_with_text, list_of_intermediate_frames_with_text)
    """
    intermediate_frames = []

    def diffusion_callback(step: int, timestep: int, latents: torch.FloatTensor):
        # (uncomment the next line for debugging)
        # print(f"Callback at step {step}, timestep {timestep}")
        with torch.no_grad():
            scaled_latents = latents / pipe.vae.config.scaling_factor
            image_tensor = pipe.vae.decode(scaled_latents).sample[
                0
            ]  # shape: (C,H,W) in [-1,1]
            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)  # convert to [0,1]
            image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_with_text = add_text_to_frame(image, prompt)
            intermediate_frames.append(image_with_text)

    # Suppress deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback=diffusion_callback,
            callback_steps=1,
        )

    # Convert final image from PIL to NumPy array and add text
    final_image_np = np.array(result.images[0])
    final_image_with_text = add_text_to_frame(final_image_np, prompt)

    return final_image_with_text, intermediate_frames


# Crossfade between two images


def crossfade_images(
    image1: np.ndarray, image2: np.ndarray, num_transition_frames: int
):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    frames = []
    for i in range(num_transition_frames):
        alpha = i / (num_transition_frames - 1)
        blended = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        frames.append(blended)
    return frames


# Initialize Stable Diffusion Pipeline (v2.1)

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="scheduler"
)

# Video Settings
fps = 30
transition_duration = 1  # seconds for crossfade
num_transition_frames = int(fps * transition_duration)

# Generate Images & Diffusion Frames for Each Prompt
final_images = []  # List of tuples: (final_image, duration, diffusion_frames)
for phrase in phrases_with_emotion:
    prompt = phrase["text"]
    duration = phrase["end"] - phrase["start"]
    print(f"Processing prompt: '{prompt}' (duration: {duration:.2f} seconds)")

    # Generate final image and capture the real intermediate diffusion frames
    final_img_with_text, diffusion_frames = generate_final_and_diffusion_frames(
        pipe, prompt, num_inference_steps=30, guidance_scale=8.5
    )

    final_images.append((final_img_with_text, duration, diffusion_frames))


# Build the Video
video_frames = []
for i, (final_img, duration, diffusion_frames) in enumerate(final_images):
    # Show the captured intermediate frames (diffusion process)
    # Hold each intermediate frame for 0.2 seconds (adjust as needed)
    for frame in diffusion_frames:
        for _ in range(int(0.2 * fps)):
            video_frames.append(frame)

    # Crossfade from the last intermediate diffusion frame to the final image
    if diffusion_frames:
        video_frames.extend(
            crossfade_images(diffusion_frames[-1], final_img, num_transition_frames)
        )

    # Hold the final image for the duration of the prompt
    hold_frames = int(duration * fps)
    for _ in range(hold_frames):
        video_frames.append(final_img)

    # Crossfade to the next prompt (if available)
    if i < len(final_images) - 1:
        next_img, _, next_diff_frames = final_images[i + 1]
        target_img = next_diff_frames[0] if next_diff_frames else next_img
        video_frames.extend(
            crossfade_images(final_img, target_img, num_transition_frames)
        )

# Create full video
frames_to_video(video_frames, output_video="video.mp4", fps=fps)
