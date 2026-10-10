import numpy as np
import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from src.utils.logger import logger
from src.utils.config_loader import cfg 

class ActionRecognizer:
    """
    Phase 2: The "Muscle" (Action Recognition).
    Responsibility: 
    1. Takes a short video clip (8 - 16 frames) of a specific person.
    2. Uses OpenCLIP to classify the action (e.g., "punching", "falling").
    """
    def __init__(self):
        desired_device = cfg['action'].get('device', cfg['system'].get('device', 'cpu'))
        if desired_device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested for ActionRecognizer but not available. Falling back to CPU.")
            self.device = 'cpu'
        else:
            self.device = desired_device
        
        # 1. Load Settings from Config
        model_name = cfg['action'].get('model_name', 'coca_ViT-L-14')
        pretrained = cfg['action'].get('weights_path', 'models/coca_l14.bin')

        logger.info(f"Loading Phase 2 Model: {model_name} on {self.device}...")
        logger.info(f"Using weights from: {pretrained}")
        
        try:
            # 2. Initialize Model
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            
            # Convert model to FP16 (Half Precision) if using GPU
            self.use_half = cfg['action'].get('half_precision', True) if ('cuda' in str(self.device)) else False
            if self.use_half:
                self.model = self.model.half()
                
            self.model.eval() # Freeze the model (Save memory)
            logger.info("Phase 2 Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load OpenCLIP: {e}")
            raise e

        # 3. Prepare the Prompts & Pre-encode Text Vectors (Once)
        self.prompts = cfg['action']['prompts']
        self.text_embeddings = self._encode_text(self.prompts)
        logger.info(f"Monitoring actions: {self.prompts}")

        # 4. Memory Optimization: Prune unused text decoder and text encoder (~700 MB VRAM freed)
        if hasattr(self.model, 'text_decoder'):
            del self.model.text_decoder
        if hasattr(self.model, 'text'):
            del self.model.text
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            logger.info(f"Phase 2 memory pruned. Active VRAM: {vram_mb:.1f} MB")

    def _encode_text(self, text_list):
        """
        Converts text strings into mathematical vectors.
        Run only ONCE at startup to save performance.
        """
        with torch.no_grad():
            tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
            text_tokens = tokenizer(text_list).to(self.device)
            
            # Get the vectors
            text_features = self.model.encode_text(text_tokens)
            # Normalize them (Required for Cosine Similarity)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def get_action_score(self, frame_list):
        """
        Input: List of Numpy Images (The Video Clip)
        Output: Dictionary { "punching": 0.85, "walking": 0.10, ... }
        """
        if not frame_list:
            return None

        # 0. Fast Motion Gating: Check inter-frame pixel difference (<0.2ms on CPU)
        # If the person is stationary/idle, bypass heavy ViT-L inference completely (0% GPU)
        motion_thresh = cfg['action'].get('motion_threshold', 15.0)
        if len(frame_list) >= 2:
            first_f = frame_list[0].astype(np.float32)
            last_f = frame_list[-1].astype(np.float32)
            motion_val = float(np.mean(np.abs(last_f - first_f)))
            if motion_val < motion_thresh:
                return {"unknown_benign_activity": 1.0}

        # 1. Preprocess Images (Numpy -> Tensor)
        try:
            # Subsample to key frames (default 3) to cut inference time down to ~1.4s
            sample_count = cfg['action'].get('sample_frames', 1)
            if sample_count == 1:
                sampled = [frame_list[-1]]
            elif len(frame_list) > sample_count:
                indices = np.linspace(0, len(frame_list) - 1, sample_count, dtype=int)
                sampled = [frame_list[i] for i in indices]
            else:
                sampled = frame_list

            images = [self.preprocess(Image.fromarray(f)).unsqueeze(0) for f in sampled]
            image_batch = torch.cat(images).to(self.device)

            # Convert to FP16 if using GPU
            if getattr(self, 'use_half', False):
                image_batch = image_batch.half()
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
        
        # 2. Run the AI
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 3. Calculate Similarity (The Dot Product)
                raw_similarity = image_features @ self.text_embeddings.T   
                scaled_similarity = 100.0 * raw_similarity
                # 4. Softmax to get percentages (0.0 to 1.0)
                probs = F.softmax(scaled_similarity, dim=-1)            
                # 5. Aggregate (Take the average score across the key frames)
                avg_scores = probs.mean(dim=0).cpu().numpy()
                max_raw_sim = raw_similarity.mean(dim=0).max().item()
        finally:
            del image_batch

        # Map scores back to text labels
        result = {prompt: score for prompt, score in zip(self.prompts, avg_scores)}
        
        # --- THE OSR GATEKEEPER ---
        # If the highest raw similarity is too low, the model is guessing blindly.
        # We override the result and force it into a safe "unknown" state.
        osr_thresh = cfg['action'].get('osr_threshold', cfg['action'].get('ors_threshold', 0.25))
        if max_raw_sim < osr_thresh:
            return {"unknown_benign_activity": 1.0}

        return result