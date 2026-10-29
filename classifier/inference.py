"""
Inference and Explainability (Grad-CAM) service for the Skin Lesion Classification project.

Architecture confirmed by model inspection:
  Input → Rescaling(scale=2.0, offset=-1.0) → MobileNetV2 → GlobalAvgPool → Dropout → Dense(7, softmax)

Preprocessing contract:
  - Resize image to 128 × 128 pixels (RGB)
  - Convert pixel values to float32 and divide by 255  →  values in [0, 1]
  - The model's internal Rescaling layer maps [0, 1] → [-1, 1] automatically
  - DO NOT apply any further normalisation (no double-scaling)
  - Add batch dimension: shape must be (1, 128, 128, 3) before predict()

Class-index mapping (alphabetical, as produced by flow_from_directory):
  0 → akiec
  1 → bcc
  2 → bkl
  3 → df
  4 → mel
  5 → nv
  6 → vasc

Grad-CAM contract:
  - Uses the last convolutional layer (Conv_1) of the MobileNetV2 backbone.
  - Computes gradients of the target class score w.r.t. the conv feature map.
  - Global average pools gradients over spatial dimensions to obtain channel importance weights.
  - Computes weighted combination of feature maps, applies ReLU, and normalizes to [0, 1].
  - Heatmap is upsampled to match the exact dimensions of the original uploaded image.
  - Generates 'jet' colormap and blended overlay with configurable transparency (alpha).
  - All images are converted to in-memory Base64 data URLs (zero permanent disk storage).
  - Handles errors gracefully without breaking ordinary predictions.
"""

import io
import base64
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class index → label mapping  (verified: alphabetical order from training)
# ---------------------------------------------------------------------------
CLASS_INDEX_MAP = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# ---------------------------------------------------------------------------
# Full class metadata
# ---------------------------------------------------------------------------
CLASS_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses & Intraepithelial Carcinoma',
        'short': 'Actinic Keratoses',
        'code': 'akiec',
        'description': (
            'Actinic keratoses are rough, scaly patches caused by years of sun exposure. '
            'They can sometimes develop into squamous cell carcinoma if left untreated.'
        ),
        'risk_level': 'high',
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'short': 'Basal Cell Carcinoma',
        'code': 'bcc',
        'description': (
            'The most common form of skin cancer. It rarely spreads to other parts of the body '
            'but can be locally destructive if left untreated. Often appears as a pearly or waxy bump.'
        ),
        'risk_level': 'high',
    },
    'bkl': {
        'name': 'Benign Keratosis-like Lesions',
        'short': 'Benign Keratosis',
        'code': 'bkl',
        'description': (
            'Includes seborrheic keratoses, solar lentigines, and lichen-planus like keratoses. '
            'These are generally non-cancerous skin growths that are very common with age.'
        ),
        'risk_level': 'low',
    },
    'df': {
        'name': 'Dermatofibroma',
        'short': 'Dermatofibroma',
        'code': 'df',
        'description': (
            'A common, benign skin growth that usually appears as a small, firm bump, '
            'most often on the legs. Generally harmless and does not require treatment.'
        ),
        'risk_level': 'low',
    },
    'mel': {
        'name': 'Melanoma',
        'short': 'Melanoma',
        'code': 'mel',
        'description': (
            'The most dangerous form of skin cancer, developing from melanocytes. '
            'Early detection is critical for successful treatment. '
            'Seek immediate dermatological evaluation if melanoma is suspected.'
        ),
        'risk_level': 'critical',
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'short': 'Melanocytic Nevi',
        'code': 'nv',
        'description': (
            'Commonly known as moles. Usually benign, but any changes in size, '
            'shape, color, or border should be evaluated promptly by a dermatologist.'
        ),
        'risk_level': 'low',
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'short': 'Vascular Lesions',
        'code': 'vasc',
        'description': (
            'Includes angiomas, angiokeratomas, and pyogenic granulomas. '
            'Usually benign, but some types can occasionally be associated with systemic conditions.'
        ),
        'risk_level': 'low',
    },
}

# ---------------------------------------------------------------------------
# Model singletons  (loaded once per Django process, reused for requests)
# ---------------------------------------------------------------------------
_model = None
_model_load_error = None

# Grad-CAM cached sub-components
_conv_extractor = None
_rescaling_layer = None
_downstream_base_layers = []
_downstream_top_layers = []
_target_conv_layer_name = 'Conv_1'


def _init_gradcam_components():
    """
    Initializes and caches the sub-model and layer references required for Grad-CAM.
    Locates the last convolutional layer (Conv_1) in MobileNetV2 and
    maps downstream layers to complete gradient backpropagation.
    """
    global _conv_extractor, _rescaling_layer
    global _downstream_base_layers, _downstream_top_layers, _target_conv_layer_name

    import tensorflow as tf

    if _model is None:
        return

    try:
        # 1. Locate the internal Rescaling layer (maps [0, 1] -> [-1, 1])
        _rescaling_layer = None
        for layer in _model.layers:
            if isinstance(layer, tf.keras.layers.Rescaling) or 'rescaling' in layer.name.lower():
                _rescaling_layer = layer
                break

        # 2. Locate the MobileNetV2 base model (nested Functional model)
        base_model = None
        base_model_idx = -1
        for idx, layer in enumerate(_model.layers):
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                base_model_idx = idx
                break

        if base_model is None:
            logger.warning("Could not find nested base model inside MobileNetV2 model.")
            return

        # 3. Locate target conv layer in MobileNetV2 (prefer 'Conv_1', else last Conv2D)
        target_layer = None
        try:
            target_layer = base_model.get_layer('Conv_1')
        except Exception:
            conv_layers = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
            if conv_layers:
                target_layer = conv_layers[-1]

        if target_layer is None:
            logger.warning("Could not identify convolutional layer in MobileNetV2 for Grad-CAM.")
            return

        _target_conv_layer_name = target_layer.name
        target_idx = base_model.layers.index(target_layer)

        # 4. Downstream layers within MobileNetV2 after Conv_1 (Conv_1_bn, out_relu)
        _downstream_base_layers = base_model.layers[target_idx + 1:]

        # 5. Downstream layers in the top model after MobileNetV2 (GAP, Dropout, Dense)
        _downstream_top_layers = _model.layers[base_model_idx + 1:]

        # 6. Feature extractor: base_model input -> target conv layer output
        _conv_extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=target_layer.output,
            name='gradcam_conv_extractor',
        )

        logger.info(
            "Grad-CAM initialized with target conv layer: '%s' (downstream base: %d, top: %d)",
            _target_conv_layer_name,
            len(_downstream_base_layers),
            len(_downstream_top_layers),
        )

    except Exception as exc:
        logger.exception("Failed to initialize Grad-CAM components: %s", exc)
        _conv_extractor = None


def _load_model_once():
    """
    Loads the Keras model from disk into the module-level singleton.
    Called automatically on the first prediction request (lazy-load).
    """
    global _model, _model_load_error

    if _model is not None or _model_load_error is not None:
        return  # already attempted

    # Suppress TensorFlow startup noise
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    try:
        import tensorflow as tf
        from django.conf import settings

        model_path = str(settings.ML_MODEL_PATH)
        if not os.path.exists(model_path):
            msg = (
                f"Model file not found: {model_path}\n"
                "Place 'ham10000_mobilenetv2.keras' inside the ml_models/ directory."
            )
            logger.error(msg)
            _model_load_error = msg
            return

        logger.info("Loading Keras model from: %s", model_path)
        _model = tf.keras.models.load_model(model_path)

        # Sanity-check: confirm input and output shapes
        expected_input = (None, 128, 128, 3)
        expected_output = (None, 7)
        actual_input = tuple(_model.input_shape)
        actual_output = tuple(_model.output_shape)

        if actual_input != expected_input:
            raise ValueError(
                f"Unexpected model input shape {actual_input}. Expected {expected_input}."
            )
        if actual_output != expected_output:
            raise ValueError(
                f"Unexpected model output shape {actual_output}. Expected {expected_output}."
            )

        logger.info(
            "Model loaded successfully. Input: %s  Output: %s",
            actual_input, actual_output,
        )

        # Initialize Grad-CAM components
        _init_gradcam_components()

    except Exception as exc:
        _model_load_error = str(exc)
        _model = None
        logger.exception("Failed to load model: %s", exc)


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def _load_pil_image(image_input):
    """
    Safely unpackages and converts a PIL Image, file-like object, or raw bytes
    into an RGB PIL Image.
    """
    from PIL import Image

    if isinstance(image_input, Image.Image):
        img = image_input.copy()
    elif hasattr(image_input, 'seek') and hasattr(image_input, 'read'):
        image_input.seek(0)
        raw = image_input.read()
        img = Image.open(io.BytesIO(raw))
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    return img.convert('RGB')


def _preprocess_image(image_input):
    """
    Preprocesses a PIL Image, file-like object, or raw bytes into the
    (1, 128, 128, 3) float32 tensor expected by the model.

    Steps:
      1. Open / convert image to RGB (handles RGBA, grayscale, palette)
      2. Resize to 128 × 128 using LANCZOS resampling
      3. Convert to NumPy float32 array
      4. Scale pixel values to [0, 1] by dividing by 255
         (The model's internal Rescaling layer then maps [0, 1] to [-1, 1])
      5. Add batch dimension → shape (1, 128, 128, 3)

    Returns:
        np.ndarray of shape (1, 128, 128, 3), dtype float32
    """
    from PIL import Image

    img = _load_pil_image(image_input)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float32)  # shape (128, 128, 3), range [0, 255]
    arr = arr / 255.0                      # scale to [0, 1]
    arr = np.expand_dims(arr, axis=0)      # shape (1, 128, 128, 3)

    return arr


def _array_to_data_url(img_array):
    """
    Encodes an uint8 RGB NumPy array into an in-memory Base64 PNG data URL.
    Ensures zero disk writes.
    """
    from PIL import Image
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG', optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Grad-CAM Explainability
# ---------------------------------------------------------------------------

def generate_gradcam(image_input, target_class_index=None, alpha=0.5):
    """
    Generates a Grad-CAM heatmap and blended overlay for a skin lesion image.

    Requirements addressed:
      - Uses TensorFlow/Keras and MobileNetV2's final conv layer (Conv_1)
      - Computes gradients of the target class score w.r.t. the conv feature map
      - Global-average-pools gradients into channel importance weights
      - Computes weighted combination of feature maps and applies ReLU
      - Upsamples heatmap to match EXACT original uploaded image dimensions
      - Applies 'jet' colormap
      - Blends heatmap onto original image with specified alpha transparency
      - Converts images to base64 data URLs in-memory (no permanent disk storage)
      - Fails gracefully without raising uncaught exceptions

    Args:
        image_input: PIL Image or file-like object containing the uploaded image.
        target_class_index: int (0..6), index of class to explain. If None, uses top class.
        alpha: float (0.0 to 1.0), blending transparency factor (default 0.5).

    Returns:
        dict with keys:
            success (bool),
            target_class_idx (int),
            target_class_code (str),
            target_layer (str),
            heatmap_data_url (str),
            overlay_data_url (str),
            original_dimensions (tuple [w, h]),
            heatmap_dimensions (tuple [w, h]),
            overlay_dimensions (tuple [w, h]),
            alpha (float),
            error (str or None)
    """
    _load_model_once()

    if _model_load_error:
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': CLASS_INDEX_MAP[target_class_index] if (target_class_index is not None and 0 <= target_class_index < 7) else None,
            'target_layer': None,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': None,
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': f"Model unavailable: {_model_load_error}",
        }

    if _conv_extractor is None:
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': None,
            'target_layer': None,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': None,
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': "Grad-CAM feature extractor not initialized.",
        }

    import tensorflow as tf
    import matplotlib
    from PIL import Image

    # 1. Obtain PIL Image and record original dimensions
    try:
        pil_image_rgb = _load_pil_image(image_input)
        orig_w, orig_h = pil_image_rgb.size
    except Exception as exc:
        logger.exception("Failed to load original image for Grad-CAM: %s", exc)
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': None,
            'target_layer': _target_conv_layer_name,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': None,
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': f"Image loading error: {exc}",
        }

    # 2. Preprocess to (1, 128, 128, 3) float32 in [0, 1]
    try:
        img_array = _preprocess_image(pil_image_rgb)
        tensor_in = tf.convert_to_tensor(img_array, dtype=tf.float32)
    except Exception as exc:
        logger.exception("Image preprocessing for Grad-CAM failed: %s", exc)
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': None,
            'target_layer': _target_conv_layer_name,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': (orig_w, orig_h),
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': f"Preprocessing error: {exc}",
        }

    # 3. Compute gradients of class score w.r.t. conv_features
    try:
        with tf.GradientTape() as tape:
            x_rescaled = _rescaling_layer(tensor_in) if _rescaling_layer is not None else tensor_in
            conv_features = _conv_extractor(x_rescaled)
            tape.watch(conv_features)

            # Pass through downstream base layers
            out = conv_features
            for layer in _downstream_base_layers:
                out = layer(out)

            # Pass through downstream top layers
            dense_logits = None
            for layer in _downstream_top_layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    out = layer(out, training=False)
                elif isinstance(layer, tf.keras.layers.Dense) and layer == _downstream_top_layers[-1]:
                    dense_logits = tf.matmul(out, layer.kernel) + layer.bias
                    out = layer.activation(dense_logits)
                else:
                    out = layer(out)

            if target_class_index is None:
                target_class_index = int(tf.argmax(out[0]))

            if not (0 <= target_class_index < 7):
                raise ValueError(f"Invalid target_class_index: {target_class_index} (must be 0..6)")

            target_code = CLASS_INDEX_MAP[target_class_index]
            score = dense_logits[:, target_class_index] if dense_logits is not None else out[:, target_class_index]

        grads = tape.gradient(score, conv_features)
        if grads is None:
            raise RuntimeError("Gradient computation returned None.")

        # 4. Grad-CAM calculation
        # Global average pooling of gradients across spatial dimensions
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)

        # Weighted combination of feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_features[0]), axis=-1)  # shape (4, 4)

        # ReLU to keep only positive contributions
        cam = tf.maximum(cam, 0)

        # Normalize to [0, 1]
        cam_max = tf.reduce_max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        cam_np = cam.numpy()

    except Exception as exc:
        logger.exception("Grad-CAM gradient computation failed: %s", exc)
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': CLASS_INDEX_MAP[target_class_index] if (target_class_index is not None and 0 <= target_class_index < 7) else None,
            'target_layer': _target_conv_layer_name,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': (orig_w, orig_h),
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': f"Grad-CAM computation error: {exc}",
        }

    # 5. Upsample heatmap to match original image dimensions (orig_w, orig_h)
    try:
        cam_uint8 = np.uint8(np.clip(255.0 * cam_np, 0, 255))
        heatmap_pil = Image.fromarray(cam_uint8).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_norm = np.array(heatmap_pil, dtype=np.float32) / 255.0

        # 6. Apply 'jet' colormap
        cmap = matplotlib.colormaps.get('jet', matplotlib.colormaps['viridis'])
        colored_heatmap = cmap(heatmap_norm)[:, :, :3]  # RGB in [0.0, 1.0]
        colored_heatmap_uint8 = np.uint8(np.clip(255.0 * colored_heatmap, 0, 255))

        # 7. Create superimposed overlay with specified alpha transparency
        orig_rgb_np = np.array(pil_image_rgb, dtype=np.float32)
        alpha_clamped = max(0.0, min(1.0, float(alpha)))
        overlay_np = np.uint8(np.clip(
            alpha_clamped * colored_heatmap_uint8 + (1.0 - alpha_clamped) * orig_rgb_np,
            0, 255
        ))

        # 8. Encode to in-memory Base64 PNG data URLs
        heatmap_data_url = _array_to_data_url(colored_heatmap_uint8)
        overlay_data_url = _array_to_data_url(overlay_np)

        return {
            'success': True,
            'target_class_idx': target_class_index,
            'target_class_code': target_code,
            'target_layer': _target_conv_layer_name,
            'heatmap_data_url': heatmap_data_url,
            'overlay_data_url': overlay_data_url,
            'original_dimensions': (orig_w, orig_h),
            'heatmap_dimensions': (orig_w, orig_h),
            'overlay_dimensions': (orig_w, orig_h),
            'alpha': alpha_clamped,
            'error': None,
        }

    except Exception as exc:
        logger.exception("Heatmap rendering or overlay generation failed: %s", exc)
        return {
            'success': False,
            'target_class_idx': target_class_index,
            'target_class_code': CLASS_INDEX_MAP[target_class_index] if (target_class_index is not None and 0 <= target_class_index < 7) else None,
            'target_layer': _target_conv_layer_name,
            'heatmap_data_url': None,
            'overlay_data_url': None,
            'original_dimensions': (orig_w, orig_h),
            'heatmap_dimensions': None,
            'overlay_dimensions': None,
            'alpha': alpha,
            'error': f"Rendering error: {exc}",
        }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(image_file, include_gradcam=True, gradcam_alpha=0.5):
    """
    Main prediction entry point called by Django views.

    Accepts: a file-like object (Django InMemoryUploadedFile or BytesIO) or PIL Image.
    Returns: a dict with the following keys:
        predicted_class       str   — class code, e.g. 'nv'
        predicted_label       str   — full name
        predicted_short       str   — abbreviated name
        predicted_prob        float — probability of top class (0–1)
        predicted_prob_pct    float — same, as percentage (0–100)
        predicted_risk        str   — 'low' | 'high' | 'critical'
        predicted_description str   — clinical description
        all_probabilities     list  — 7 dicts sorted by probability desc
        gradcam               dict  — Grad-CAM explainability data (or None/error dict)
        is_mock               bool  — False
        error                 str|None
    """
    # Ensure model is loaded
    _load_model_once()

    if _model_load_error:
        return _error_result(
            f"Model could not be loaded: {_model_load_error}. "
            "Please ensure 'ml_models/ham10000_mobilenetv2.keras' exists."
        )

    # Open PIL Image to preserve original resolution for Grad-CAM
    try:
        pil_image = _load_pil_image(image_file)
    except Exception as exc:
        logger.exception("Image loading failed: %s", exc)
        return _error_result(
            f"Could not process the uploaded image: {exc}. "
            "Please ensure it is a valid, uncorrupted JPG or PNG file."
        )

    # Preprocess image for inference: (1, 128, 128, 3) float32 in [0, 1]
    try:
        img_array = _preprocess_image(pil_image)
    except Exception as exc:
        logger.exception("Image preprocessing failed: %s", exc)
        return _error_result(
            f"Could not preprocess the uploaded image: {exc}."
        )

    # Run model inference
    try:
        raw_preds = _model.predict(img_array, verbose=0)  # shape (1, 7)
    except Exception as exc:
        logger.exception("Model inference failed: %s", exc)
        return _error_result(f"Inference error: {exc}")

    probs = raw_preds[0].tolist()  # list of 7 floats, sum ≈ 1.0

    if len(probs) != 7:
        return _error_result(
            f"Unexpected number of outputs: {len(probs)} (expected 7)."
        )

    if not all(np.isfinite(p) for p in probs):
        return _error_result(
            "Model produced non-finite probability scores."
        )

    # Map index → class code using the verified alphabetical mapping
    class_prob_pairs = [
        (CLASS_INDEX_MAP[i], float(probs[i]))
        for i in range(7)
    ]

    # Sort descending by probability
    class_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    predicted_code, predicted_prob = class_prob_pairs[0]
    info = CLASS_INFO[predicted_code]

    all_probabilities = [
        {
            'code': code,
            'label': CLASS_INFO[code]['name'],
            'short': CLASS_INFO[code]['short'],
            'probability': prob,
            'probability_pct': round(prob * 100, 2),
            'risk_level': CLASS_INFO[code]['risk_level'],
        }
        for code, prob in class_prob_pairs
    ]

    logger.info(
        "Prediction: class=%s prob=%.4f (raw probs by index: %s)",
        predicted_code,
        predicted_prob,
        [round(p, 4) for p in probs],
    )

    # Generate Grad-CAM explainability for the top predicted class
    gradcam_result = None
    if include_gradcam:
        try:
            top_class_idx = CLASS_INDEX_MAP.index(predicted_code)
            gradcam_result = generate_gradcam(
                pil_image,
                target_class_index=top_class_idx,
                alpha=gradcam_alpha,
            )
        except Exception as exc:
            # Graceful error handling: Grad-CAM failure does NOT break the prediction
            logger.exception("Grad-CAM generation failed: %s", exc)
            gradcam_result = {
                'success': False,
                'target_class_idx': CLASS_INDEX_MAP.index(predicted_code) if predicted_code in CLASS_INDEX_MAP else None,
                'target_class_code': predicted_code,
                'target_layer': _target_conv_layer_name,
                'heatmap_data_url': None,
                'overlay_data_url': None,
                'original_dimensions': pil_image.size if pil_image else None,
                'heatmap_dimensions': None,
                'overlay_dimensions': None,
                'alpha': gradcam_alpha,
                'error': str(exc),
            }

    return {
        'predicted_class':       predicted_code,
        'predicted_label':       info['name'],
        'predicted_short':       info['short'],
        'predicted_prob':        predicted_prob,
        'predicted_prob_pct':    round(predicted_prob * 100, 2),
        'predicted_risk':        info['risk_level'],
        'predicted_description': info['description'],
        'all_probabilities':     all_probabilities,
        'gradcam':               gradcam_result,
        'is_mock':               False,
        'error':                 None,
    }


def _error_result(message):
    """Returns a safe error-state result dict."""
    return {
        'predicted_class':       None,
        'predicted_label':       'Error',
        'predicted_short':       'Error',
        'predicted_prob':        0.0,
        'predicted_prob_pct':    0.0,
        'predicted_risk':        'low',
        'predicted_description': '',
        'all_probabilities':     [],
        'gradcam':               None,
        'is_mock':               False,
        'error':                 message,
    }


def get_model_status():
    """
    Returns a tuple (loaded: bool, error: str|None).
    Used by views to show server-side model status messages.
    """
    _load_model_once()
    return (_model is not None), _model_load_error
