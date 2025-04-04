import os
import sys
import torch
import numpy as np
from collections.abc import Mapping
import math

class DJ_VideoAudioMixer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images1": ("IMAGE", ),
                "video_info1": ("VHS_VIDEOINFO", ),
            },
            "optional": {
                "audio1": ("AUDIO", ),
                "images2": ("IMAGE", ),
                "audio2": ("AUDIO", ),
                "video_info2": ("VHS_VIDEOINFO", ),
                "bgm": ("AUDIO", ),
                "bgm_mode": (["all", "first_video", "second_video"], {"default": "all"}),
                "bgm_volume": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fade_in_sec": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "fade_out_sec": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "audio_match_method": (["pad_with_silence", "repeat_audio"], {"default": "pad_with_silence"}),
            }
        }

    CATEGORY = "audio/video processing"
    FUNCTION = "VideoAudioMixer"
    RETURN_NAMES = ("images_output", "audio_output", "video_info_output")
    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_VIDEOINFO")

    def sanitize_audio(self, audio_tensor, label="audio"):
        """
        Aggressively clean audio data to ensure it contains no invalid values
        """
        # Check for NaN or Inf before processing
        has_nan = torch.isnan(audio_tensor).any()
        has_inf = torch.isinf(audio_tensor).any()
        
        if has_nan or has_inf:
            print(f"WARNING: Found NaN or Inf in {label} before sanitization")
            
        # Replace NaN and infinity values
        audio_tensor = torch.nan_to_num(audio_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for extreme values
        max_val = audio_tensor.abs().max().item()
        if max_val > 10.0:
            print(f"WARNING: Extreme values detected in {label}: {max_val}")
        
        # First pass clipping for extreme outliers
        audio_tensor = torch.clamp(audio_tensor, min=-2.0, max=2.0)
        
        # Apply a smoother limiter to avoid harsh clipping
        max_amplitude = audio_tensor.abs().max()
        if max_amplitude > 0.95:
            # Calculate a softer reduction ratio
            ratio = 0.95 / max_amplitude
            audio_tensor = audio_tensor * ratio
            print(f"Applied soft limiter with ratio {ratio:.4f} to {label}")
        
        # Final safety clamp to ensure values are in valid range
        audio_tensor = torch.clamp(audio_tensor, min=-0.95, max=0.95)
        
        return audio_tensor

    def VideoAudioMixer(self, images1, video_info1, audio1=None, images2=None, audio2=None, video_info2=None, 
                        bgm=None, bgm_mode="all", bgm_volume=0.3, fade_in_sec=1.0, fade_out_sec=1.0,
                        audio_match_method="pad_with_silence"):
        if images2 is None or video_info2 is None:
            # Return with just the first video and its audio (if any)
            return (images1, audio1, video_info1)
            
        print(f"DEBUG: bgm={bgm is not None}, bgm_mode={bgm_mode}, bgm_volume={bgm_volume}, fade_in_sec={fade_in_sec}, fade_out_sec={fade_out_sec}")

        # Verify frames
        if not isinstance(images1, torch.Tensor) or not isinstance(images2, torch.Tensor):
            raise ValueError("images1 and images2 must be frame tensors")

        # Handle frame dimensions (assume [frames, h, w, c])
        if images1.shape[1:] != images2.shape[1:]:
            raise ValueError(f"Incompatible resolutions: images1 {images1.shape}, images2 {images2.shape}")

        # Concatenate frames
        concatenated_frames = torch.cat([images1, images2], dim=0)  # [frames_total, h, w, c]

        # Extract FPS from video_info
        fps1 = video_info1["loaded_fps"]
        fps2 = video_info2["loaded_fps"]
        if fps1 != fps2:
            print(f"Warning: Different FPS (video1: {fps1}, video2: {fps2}), using {fps1}")
        output_fps = fps1

        # Handle audio
        audio_output = None
        sample_rate = None

        # Function to extract waveform and sample_rate from audio
        def get_audio_data(audio_input, label=""):
            print(f"{label}: Type of audio_input = {type(audio_input)}, value = {audio_input}")
            if audio_input is None:
                print(f"{label}: No audio provided")
                return None, None
            
            if isinstance(audio_input, Mapping):
                try:
                    waveform = audio_input["waveform"].squeeze(0)
                    sample_rate = audio_input["sample_rate"]
                    print(f"{label}: Audio extracted from Mapping, waveform shape={waveform.shape}, sample_rate={sample_rate}")
                    return waveform, sample_rate
                except KeyError as e:
                    print(f"{label}: Error - Missing key in Mapping: {e}")
                    return None, None
                except Exception as e:
                    print(f"{label}: Error extracting from Mapping: {e}")
                    return None, None
            
            elif callable(audio_input):
                try:
                    audio_data = audio_input()
                    if isinstance(audio_data, dict) and "waveform" in audio_data:
                        waveform = audio_data["waveform"].squeeze(0)
                        print(f"{label}: Audio extracted from function, waveform shape={waveform.shape}, sample_rate={audio_data['sample_rate']}")
                        return waveform, audio_data["sample_rate"]
                    else:
                        print(f"{label}: Invalid function result: {audio_data}")
                        return None, None
                except Exception as e:
                    print(f"{label}: Error evaluating function: {e}")
                    return None, None
            
            elif isinstance(audio_input, dict) and "waveform" in audio_input:
                waveform = audio_input["waveform"].squeeze(0)
                print(f"{label}: Audio extracted from dictionary, waveform shape={waveform.shape}, sample_rate={audio_input['sample_rate']}")
                return waveform, audio_input["sample_rate"]
            
            else:
                print(f"{label}: Audio format not recognized: {type(audio_input)}")
                return None, None

        # Extract audio data from all audio inputs
        audio_waveform1, sample_rate1 = get_audio_data(audio1, "Audio1")
        audio_waveform2, sample_rate2 = get_audio_data(audio2, "Audio2")
        bgm_waveform, bgm_sample_rate = get_audio_data(bgm, "BGM")

        # Apply early sanitization
        if audio_waveform1 is not None:
            audio_waveform1 = self.sanitize_audio(audio_waveform1, "audio1 (initial)")
        if audio_waveform2 is not None:
            audio_waveform2 = self.sanitize_audio(audio_waveform2, "audio2 (initial)")
        if bgm_waveform is not None:
            bgm_waveform = self.sanitize_audio(bgm_waveform, "bgm (initial)")

        # Determine target sample rate (prefer audio1, then audio2, then BGM, fallback to 44100)
        sample_rate = sample_rate1 or sample_rate2 or bgm_sample_rate or 44100  
        print(f"Using sample rate: {sample_rate}Hz")

        try:
            # Import torchaudio for resampling
            import torchaudio

            # Resample all audio to the target sample rate
            if audio_waveform1 is not None and sample_rate1 != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate1,
                    new_freq=sample_rate
                )
                audio_waveform1 = resampler(audio_waveform1)
                audio_waveform1 = self.sanitize_audio(audio_waveform1, "audio1 (resampled)")
                print(f"Resampled audio1 from {sample_rate1}Hz to {sample_rate}Hz")
            
            if audio_waveform2 is not None and sample_rate2 != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate2,
                    new_freq=sample_rate
                )
                audio_waveform2 = resampler(audio_waveform2)
                audio_waveform2 = self.sanitize_audio(audio_waveform2, "audio2 (resampled)")
                print(f"Resampled audio2 from {sample_rate2}Hz to {sample_rate}Hz")
            
            if bgm_waveform is not None and bgm_sample_rate != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=bgm_sample_rate,
                    new_freq=sample_rate
                )
                bgm_waveform = resampler(bgm_waveform)
                bgm_waveform = self.sanitize_audio(bgm_waveform, "bgm (resampled)")
                print(f"Resampled BGM from {bgm_sample_rate}Hz to {sample_rate}Hz")
        except Exception as e:
            print(f"Error during resampling: {e}. Using original sample rates.")

        # Calculate durations
        duration1 = images1.shape[0] / fps1
        duration2 = images2.shape[0] / fps2
        total_duration = duration1 + duration2
        total_samples = int(total_duration * sample_rate)
        samples1 = int(duration1 * sample_rate)
        samples2 = int(duration2 * sample_rate)

        print(f"Video durations: Video1={duration1:.2f}s, Video2={duration2:.2f}s, Total={total_duration:.2f}s")
        print(f"Audio samples: Audio1={samples1}, Audio2={samples2}, Total={total_samples}")

        # Generate silent audio for missing primary audio inputs
        if audio_waveform1 is None:
            audio_waveform1 = torch.zeros((1, samples1))
            print(f"Audio1: Silence generated, shape={audio_waveform1.shape}")
        elif audio_waveform1.shape[1] < samples1:
            # Audio is shorter than video
            if audio_match_method == "repeat_audio":
                # Calculate how many times to repeat the audio
                repeats_needed = math.ceil(samples1 / audio_waveform1.shape[1])
                repeated_audio = audio_waveform1.repeat(1, repeats_needed)
                # Trim to exact length needed
                audio_waveform1 = repeated_audio[:, :samples1]
                print(f"Audio1 repeated {repeats_needed} times to match video duration")
            else:  # pad_with_silence
                # Pad with silence
                padding = torch.zeros(audio_waveform1.shape[0], samples1 - audio_waveform1.shape[1])
                audio_waveform1 = torch.cat([audio_waveform1, padding], dim=1)
                print(f"Audio1 padded with silence to match video duration")

        if audio_waveform2 is None:
            audio_waveform2 = torch.zeros((1, samples2))
            print(f"Audio2: Silence generated, shape={audio_waveform2.shape}")
        elif audio_waveform2.shape[1] < samples2:
            # Audio is shorter than video
            if audio_match_method == "repeat_audio":
                # Calculate how many times to repeat the audio
                repeats_needed = math.ceil(samples2 / audio_waveform2.shape[1])
                repeated_audio = audio_waveform2.repeat(1, repeats_needed)
                # Trim to exact length needed
                audio_waveform2 = repeated_audio[:, :samples2]
                print(f"Audio2 repeated {repeats_needed} times to match video duration")
            else:  # pad_with_silence
                # Pad with silence
                padding = torch.zeros(audio_waveform2.shape[0], samples2 - audio_waveform2.shape[1])
                audio_waveform2 = torch.cat([audio_waveform2, padding], dim=1)
                print(f"Audio2 padded with silence to match video duration")

        # Match channel counts between primary audio streams
        if audio_waveform1.shape[0] != audio_waveform2.shape[0]:
            print(f"Channel count mismatch: audio1 has {audio_waveform1.shape[0]} channels, audio2 has {audio_waveform2.shape[0]} channels")
            
            # If audio1 is mono and audio2 is stereo
            if audio_waveform1.shape[0] == 1 and audio_waveform2.shape[0] == 2:
                # Convert audio1 to stereo by duplicating the channel
                audio_waveform1 = audio_waveform1.repeat(2, 1)
                print(f"Converted audio1 to stereo: new shape={audio_waveform1.shape}")
                
            # If audio1 is stereo and audio2 is mono
            elif audio_waveform1.shape[0] == 2 and audio_waveform2.shape[0] == 1:
                # Convert audio2 to stereo by duplicating the channel
                audio_waveform2 = audio_waveform2.repeat(2, 1)
                print(f"Converted audio2 to stereo: new shape={audio_waveform2.shape}")

        # Concatenate the primary audio streams
        primary_audio = torch.cat([audio_waveform1, audio_waveform2], dim=1)
        primary_audio = self.sanitize_audio(primary_audio, "primary_audio (concatenated)")
        print(f"Concatenated primary audio: shape={primary_audio.shape}")

        # Check if primary audio has actual content (not just silence)
        has_actual_audio = primary_audio.abs().max() > 0.01
        print(f"Primary audio has actual content: {has_actual_audio}")

        # Process background music if provided
        if bgm_waveform is not None:
            print(f"Processing BGM with mode: {bgm_mode}")
            
            # Match channel count with primary audio
            primary_channels = primary_audio.shape[0]
            if bgm_waveform.shape[0] != primary_channels:
                if bgm_waveform.shape[0] == 1 and primary_channels == 2:
                    # Convert mono BGM to stereo
                    bgm_waveform = bgm_waveform.repeat(2, 1)
                    print(f"Converted mono BGM to stereo: new shape={bgm_waveform.shape}")
                elif bgm_waveform.shape[0] == 2 and primary_channels == 1:
                    # Convert stereo BGM to mono
                    bgm_waveform = bgm_waveform.mean(dim=0, keepdim=True)
                    print(f"Converted stereo BGM to mono: new shape={bgm_waveform.shape}")
            
            # Prepare BGM according to selected mode
            if bgm_mode == "all":
                # Apply BGM to the entire video
                target_samples = total_samples
                
                # Loop or trim BGM to match total audio length
                if bgm_waveform.shape[1] < target_samples:
                    # BGM is shorter than needed, loop it
                    repeats_needed = (target_samples + bgm_waveform.shape[1] - 1) // bgm_waveform.shape[1]
                    bgm_repeated = bgm_waveform.repeat(1, repeats_needed)
                    bgm_waveform_processed = bgm_repeated[:, :target_samples]
                    print(f"Looped BGM {repeats_needed} times to match total duration, new shape={bgm_waveform_processed.shape}")
                elif bgm_waveform.shape[1] > target_samples:
                    # BGM is longer than needed, trim it
                    bgm_waveform_processed = bgm_waveform[:, :target_samples]
                    print(f"Trimmed BGM to match total duration, new shape={bgm_waveform_processed.shape}")
                else:
                    bgm_waveform_processed = bgm_waveform
                
                # Create a mask that applies BGM to the entire audio
                bgm_mask = torch.ones(target_samples)
                
            elif bgm_mode == "first_video":
                # Apply BGM only to the first video segment
                target_samples = samples1
                
                # Loop or trim BGM to match first video length
                if bgm_waveform.shape[1] < target_samples:
                    # BGM is shorter than needed, loop it
                    repeats_needed = (target_samples + bgm_waveform.shape[1] - 1) // bgm_waveform.shape[1]
                    bgm_repeated = bgm_waveform.repeat(1, repeats_needed)
                    bgm_segment = bgm_repeated[:, :target_samples]
                    print(f"Looped BGM {repeats_needed} times to match first video duration")
                else:
                    # BGM is longer than needed, trim it
                    bgm_segment = bgm_waveform[:, :target_samples]
                    print(f"Trimmed BGM to match first video duration")
                
                # Pad with zeros for the second video segment
                zeros_pad = torch.zeros(bgm_segment.shape[0], samples2)
                bgm_waveform_processed = torch.cat([bgm_segment, zeros_pad], dim=1)
                print(f"Created BGM for first video only, shape={bgm_waveform_processed.shape}")
                
                # Create mask for fade-out at the end of the first segment
                bgm_mask = torch.ones(total_samples)
                if fade_out_sec > 0:
                    fade_out_samples = min(int(fade_out_sec * sample_rate), target_samples)
                    fade_indices = torch.arange(target_samples - fade_out_samples, target_samples)
                    fade_values = torch.linspace(1.0, 0.0, fade_out_samples)
                    bgm_mask[fade_indices] = fade_values
                    bgm_mask[target_samples:] = 0
                    print(f"Applied {fade_out_sec}s fade-out at the end of first video")
                
            elif bgm_mode == "second_video":
                # Apply BGM only to the second video segment
                target_samples = samples2
                
                # Loop or trim BGM to match second video length
                if bgm_waveform.shape[1] < target_samples:
                    # BGM is shorter than needed, loop it
                    repeats_needed = (target_samples + bgm_waveform.shape[1] - 1) // bgm_waveform.shape[1]
                    bgm_repeated = bgm_waveform.repeat(1, repeats_needed)
                    bgm_segment = bgm_repeated[:, :target_samples]
                    print(f"Looped BGM {repeats_needed} times to match second video duration")
                else:
                    # BGM is longer than needed, trim it
                    bgm_segment = bgm_waveform[:, :target_samples]
                    print(f"Trimmed BGM to match second video duration")
                
                # Pad with zeros for the first video segment
                zeros_pad = torch.zeros(bgm_segment.shape[0], samples1)
                bgm_waveform_processed = torch.cat([zeros_pad, bgm_segment], dim=1)
                print(f"Created BGM for second video only, shape={bgm_waveform_processed.shape}")
                
                # Create mask for fade-in at the start of the second segment
                bgm_mask = torch.ones(total_samples)
                bgm_mask[:samples1] = 0  # Zero out first segment
                if fade_in_sec > 0:
                    fade_in_samples = min(int(fade_in_sec * sample_rate), target_samples)
                    fade_indices = torch.arange(samples1, samples1 + fade_in_samples)
                    fade_values = torch.linspace(0.0, 1.0, fade_in_samples)
                    bgm_mask[fade_indices] = fade_values
                    print(f"Applied {fade_in_sec}s fade-in at the start of second video")
            
            # Sanitize processed BGM
            bgm_waveform_processed = self.sanitize_audio(bgm_waveform_processed, "bgm (processed)")
            
            # Apply global fade-in if needed
            if bgm_mode == "all" and fade_in_sec > 0:
                fade_samples = int(fade_in_sec * sample_rate)
                if fade_samples > 0 and fade_samples < bgm_waveform_processed.shape[1]:
                    fade_curve = torch.linspace(0, 1, fade_samples)
                    for i in range(fade_samples):
                        bgm_mask[i] *= fade_curve[i]
                    print(f"Applied {fade_in_sec}s global fade-in to BGM")
            
            # Apply global fade-out if needed
            if bgm_mode == "all" and fade_out_sec > 0:
                fade_samples = int(fade_out_sec * sample_rate)
                if fade_samples > 0 and fade_samples < bgm_waveform_processed.shape[1]:
                    end_idx = bgm_waveform_processed.shape[1]
                    start_idx = max(0, end_idx - fade_samples)
                    fade_curve = torch.linspace(1, 0, end_idx - start_idx)
                    for i in range(len(fade_curve)):
                        bgm_mask[start_idx + i] *= fade_curve[i]
                    print(f"Applied {fade_out_sec}s global fade-out to BGM")
            
            # Mix BGM with primary audio
            if has_actual_audio:
                # For speech audio, we need a smoother approach than instant volume changes
                # Use a sliding window average to detect audio presence, then smooth the volume control
                
                # Step 1: Calculate audio energy over time with a sliding window
                window_size = int(0.3 * sample_rate)  # 300ms window, good for speech
                primary_energy = torch.zeros(primary_audio.shape[1])
                
                # Calculate energy profile
                for i in range(primary_audio.shape[1]):
                    start = max(0, i - window_size//2)
                    end = min(primary_audio.shape[1], i + window_size//2)
                    window_data = primary_audio[:, start:end]
                    primary_energy[i] = window_data.abs().mean()
                
                # Step 2: Apply smoothing to the energy profile
                smoothing_window = int(0.5 * sample_rate)  # 500ms smoothing window
                smoothed_energy = torch.zeros_like(primary_energy)
                for i in range(len(primary_energy)):
                    start = max(0, i - smoothing_window//2)
                    end = min(len(primary_energy), i + smoothing_window//2)
                    smoothed_energy[i] = primary_energy[start:end].mean()
                
                # Step 3: Convert energy to volume level
                # Set threshold - below this energy level, BGM will be at full volume
                threshold = 0.005
                # Set range - how quickly it transitions from min to max volume
                range_factor = 0.01
                
                # Create the dynamic volume mask based on speech
                speech_volume_mask = torch.ones(primary_audio.shape[1])
                
                for i in range(len(smoothed_energy)):
                    # Map energy to volume: higher energy = lower BGM volume
                    energy = smoothed_energy[i]
                    if energy > threshold:
                        # Linear mapping from energy to volume
                        volume_factor = max(bgm_volume, 1.0 - (energy - threshold) / range_factor)
                        speech_volume_mask[i] = volume_factor
                    else:
                        # Below threshold, full volume
                        speech_volume_mask[i] = 1.0
                
                # Combine speech-based volume control with segment masks
                # Make sure masks are the same length before combining
                if len(speech_volume_mask) != len(bgm_mask):
                    print(f"Adjusting mask lengths - speech_mask: {len(speech_volume_mask)}, bgm_mask: {len(bgm_mask)}")
                    min_length = min(len(speech_volume_mask), len(bgm_mask))
                    speech_volume_mask = speech_volume_mask[:min_length]
                    bgm_mask = bgm_mask[:min_length]

                combined_mask = speech_volume_mask * bgm_mask
                
                # Apply the volume mask to all BGM channels
                volume_adjusted_bgm = bgm_waveform_processed.clone()
                for c in range(bgm_waveform_processed.shape[0]):
                    volume_adjusted_bgm[c, :min(bgm_waveform_processed.shape[1], len(combined_mask))] *= combined_mask[:min(bgm_waveform_processed.shape[1], len(combined_mask))]
                
                volume_adjusted_bgm = self.sanitize_audio(volume_adjusted_bgm, "volume_adjusted_bgm")
                print(f"Applied adaptive volume control with speech detection and {bgm_mode} mode")
                
                # Ensure primary_audio and BGM are the same length
                if primary_audio.shape[1] < volume_adjusted_bgm.shape[1]:
                    # Pad primary audio with zeros
                    padding = torch.zeros(primary_audio.shape[0], volume_adjusted_bgm.shape[1] - primary_audio.shape[1])
                    primary_audio = torch.cat([primary_audio, padding], dim=1)
                elif primary_audio.shape[1] > volume_adjusted_bgm.shape[1]:
                    # Pad BGM with zeros
                    padding = torch.zeros(volume_adjusted_bgm.shape[0], primary_audio.shape[1] - volume_adjusted_bgm.shape[1])
                    volume_adjusted_bgm = torch.cat([volume_adjusted_bgm, padding], dim=1)
                
                # Mix and sanitize
                audio_output = primary_audio + volume_adjusted_bgm
                audio_output = self.sanitize_audio(audio_output, "mixed_audio")
                print(f"Mixed BGM with primary audio")
            else:
                # If there's no actual audio content, use BGM at full volume with segment mask
                volume_adjusted_bgm = bgm_waveform_processed.clone()
                for c in range(bgm_waveform_processed.shape[0]):
                    volume_adjusted_bgm[c, :min(bgm_waveform_processed.shape[1], len(bgm_mask))] *= bgm_mask[:min(bgm_waveform_processed.shape[1], len(bgm_mask))]
                
                audio_output = volume_adjusted_bgm
                audio_output = self.sanitize_audio(audio_output, "bgm_as_output")
                print(f"Using BGM with {bgm_mode} mode (no primary audio content)")
        else:
            # No BGM provided, just use primary audio
            audio_output = primary_audio
            audio_output = self.sanitize_audio(audio_output, "primary_audio_as_output")
            print("No BGM provided, using only primary audio")

        # Final sanitization pass
        audio_output = self.sanitize_audio(audio_output, "final_output")

        # Ensure audio_output has correct batch dimension
        audio_output = audio_output.unsqueeze(0)

        # Update video_info for output
        video_info_output = {
            "loaded_fps": output_fps,
            "loaded_frame_count": concatenated_frames.shape[0],
            "loaded_duration": concatenated_frames.shape[0] / output_fps,
            "loaded_width": images1.shape[2],  # Width
            "loaded_height": images1.shape[1],  # Height
            "source_fps": video_info1["source_fps"],
            "source_frame_count": video_info1["source_frame_count"] + video_info2["source_frame_count"],
            "source_duration": video_info1["source_duration"] + video_info2["source_duration"],
            "source_width": video_info1["source_width"],
            "source_height": video_info1["source_height"],
        }

        # Prepare audio output
        audio_output_dict = None
        if audio_output is not None:
            audio_output_dict = {
                "waveform": audio_output,
                "sample_rate": sample_rate
            }
            print(f"Final audio prepared: waveform shape={audio_output_dict['waveform'].shape}, sample_rate={sample_rate}")
        else:
            print("No final audio generated")

        print(f"Output: frames={concatenated_frames.shape}, audio={audio_output.shape if audio_output is not None else 'none'}")

        return (concatenated_frames, audio_output_dict, video_info_output)

# This part is needed when you save this as a standalone file
NODE_CLASS_MAPPINGS = {
    "DJ_VideoAudioMixer": DJ_VideoAudioMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_VideoAudioMixer": "Video Audio Mixer 🎵"
}