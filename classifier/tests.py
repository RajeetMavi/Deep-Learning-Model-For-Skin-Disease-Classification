"""
Comprehensive test suite for Skin Lesion Classification: Phase 4 Validation.

Covers:
  1. Image upload validation (valid JPG/JPEG/PNG, invalid types, empty uploads, corrupted files, oversized files)
  2. Model inference & singleton reuse, input/output tensor shapes, class mapping, probability bounds
  3. Grad-CAM explainability, dimension matching, all 7 classes, transparency alpha, error isolation
  4. Django functionality, view routing, CSRF protection, session handling, zero disk-file leakage, log safety
"""

import io
import os
import logging
import numpy as np
from PIL import Image
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from classifier.forms import ImageUploadForm
from classifier.inference import (
    CLASS_INDEX_MAP,
    CLASS_INFO,
    predict,
    generate_gradcam,
    get_model_status,
    _load_model_once,
    _preprocess_image,
)


def _create_synthetic_lesion_image(width=600, height=450, format='JPEG'):
    """
    Creates a valid synthetic dermoscopic image with a darker pigmented lesion
    in the center on a lighter skin tone background.
    """
    img = Image.new('RGB', (width, height), color=(225, 195, 185))
    pixels = img.load()
    cx, cy = width // 2, height // 2
    rx, ry = max(4, int(width * 0.2)), max(4, int(height * 0.25))

    for x in range(cx - rx, cx + rx):
        for y in range(cy - ry, cy + ry):
            dx = (x - cx) / float(rx)
            dy = (y - cy) / float(ry)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= 1.0:
                factor = 1.0 - dist_sq
                r = int(225 - 150 * factor)
                g = int(195 - 140 * factor)
                b = int(185 - 130 * factor)
                pixels[x, y] = (max(0, r), max(0, g), max(0, b))

    buf = io.BytesIO()
    img.save(buf, format=format)
    return img, buf.getvalue()


# ==============================================================================
# 1. Model Inference & Singleton Tests
# ==============================================================================

class ModelInferenceAndSingletonTests(TestCase):
    """Verifies Keras model architecture, singleton loading, preprocessing, and probability math."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        loaded, err = get_model_status()
        assert loaded, f"Model must be available for testing. Error: {err}"

    def test_model_loads_successfully_and_is_reused(self):
        """Confirm model is loaded once and subsequent calls reuse the exact same singleton instance."""
        import classifier.inference as inf
        _load_model_once()
        model_instance_1 = inf._model
        self.assertIsNotNone(model_instance_1)

        # Call again — must reuse the exact same object in memory
        _load_model_once()
        model_instance_2 = inf._model
        self.assertIs(model_instance_1, model_instance_2, "Model singleton was not reused!")

        status_loaded, status_err = get_model_status()
        self.assertTrue(status_loaded)
        self.assertIsNone(status_err)

    def test_model_input_and_output_shapes(self):
        """Verify model input shape is (None, 128, 128, 3) and output shape is (None, 7)."""
        import classifier.inference as inf
        model = inf._model

        expected_input = (None, 128, 128, 3)
        expected_output = (None, 7)

        self.assertEqual(tuple(model.input_shape), expected_input)
        self.assertEqual(tuple(model.output_shape), expected_output)

    def test_class_index_mapping_consistency(self):
        """Confirm alphabetical class-index mapping matches training set directories exactly."""
        expected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.assertEqual(CLASS_INDEX_MAP, expected_classes)
        self.assertEqual(CLASS_INDEX_MAP, sorted(CLASS_INDEX_MAP))

        # Check metadata exists for all classes
        for code in expected_classes:
            self.assertIn(code, CLASS_INFO)
            self.assertIn('name', CLASS_INFO[code])
            self.assertIn('short', CLASS_INFO[code])
            self.assertIn('risk_level', CLASS_INFO[code])
            self.assertIn(CLASS_INFO[code]['risk_level'], ['low', 'medium', 'high', 'critical'])

    def test_preprocessing_contract(self):
        """Confirm preprocessing produces float32 tensor of shape (1, 128, 128, 3) in [0, 1]."""
        img, _ = _create_synthetic_lesion_image(width=500, height=300)
        tensor = _preprocess_image(img)

        self.assertEqual(tensor.shape, (1, 128, 128, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_predict_probabilities_properties(self):
        """Confirm predictions have 7 finite values, sum to 1.0, and are sorted descending."""
        _, img_bytes = _create_synthetic_lesion_image(width=400, height=300)
        result = predict(io.BytesIO(img_bytes))

        self.assertIsNone(result['error'])
        self.assertIn(result['predicted_class'], CLASS_INDEX_MAP)

        all_probs = result['all_probabilities']
        self.assertEqual(len(all_probs), 7)

        # Sum of probabilities must equal ~1.0
        prob_sum = sum(item['probability'] for item in all_probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # Check probability bounds and finiteness
        for item in all_probs:
            prob = item['probability']
            self.assertTrue(np.isfinite(prob), f"Probability for {item['code']} is not finite!")
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertEqual(item['probability_pct'], round(prob * 100, 2))

        # Check descending order
        prob_values = [item['probability'] for item in all_probs]
        self.assertEqual(prob_values, sorted(prob_values, reverse=True))


# ==============================================================================
# 2. Grad-CAM Explainability Tests
# ==============================================================================

class GradCAMExplainabilityTests(TestCase):
    """Verifies Grad-CAM heatmap generation, dimension matching, and defensive error handling."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        loaded, err = get_model_status()
        assert loaded, f"Model must be loaded for tests. Error: {err}"

    def test_gradcam_heatmap_and_overlay_dimensions_match_input(self):
        """Test Grad-CAM with various image resolutions and aspect ratios."""
        test_resolutions = [
            (600, 450),  # standard 4:3
            (800, 600),  # larger 4:3
            (320, 240),  # small 4:3
            (500, 500),  # square 1:1
            (720, 405),  # 16:9 widescreen
        ]

        for w, h in test_resolutions:
            with self.subTest(width=w, height=h):
                img, _ = _create_synthetic_lesion_image(width=w, height=h)
                result = generate_gradcam(img, target_class_index=5, alpha=0.5)

                self.assertTrue(result['success'], f"Grad-CAM failed for {w}x{h}: {result.get('error')}")
                self.assertIsNone(result['error'])
                self.assertEqual(result['original_dimensions'], (w, h))
                self.assertEqual(result['heatmap_dimensions'], (w, h))
                self.assertEqual(result['overlay_dimensions'], (w, h))
                self.assertTrue(result['heatmap_data_url'].startswith('data:image/png;base64,'))
                self.assertTrue(result['overlay_data_url'].startswith('data:image/png;base64,'))

    def test_gradcam_across_all_seven_classes(self):
        """Confirm Grad-CAM generates valid heatmaps for each of the 7 classes."""
        img, _ = _create_synthetic_lesion_image(width=200, height=200)

        for idx, code in enumerate(CLASS_INDEX_MAP):
            with self.subTest(class_index=idx, class_code=code):
                result = generate_gradcam(img, target_class_index=idx)
                self.assertTrue(result['success'])
                self.assertEqual(result['target_class_idx'], idx)
                self.assertEqual(result['target_class_code'], code)
                self.assertEqual(result['target_layer'], 'Conv_1')

    def test_gradcam_adjustable_transparency_alpha(self):
        """Confirm alpha parameter changes overlay blending correctly."""
        img, _ = _create_synthetic_lesion_image(width=300, height=200)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            with self.subTest(alpha=alpha):
                result = generate_gradcam(img, target_class_index=4, alpha=alpha)
                self.assertTrue(result['success'])
                self.assertEqual(result['alpha'], alpha)

    def test_gradcam_error_does_not_break_ordinary_prediction(self):
        """Confirm that if Grad-CAM fails, ordinary prediction completes successfully."""
        import classifier.inference as inf
        original_extractor = inf._conv_extractor
        try:
            # Temporarily simulate Grad-CAM component failure
            inf._conv_extractor = None

            _, img_bytes = _create_synthetic_lesion_image(width=300, height=200)
            res = predict(io.BytesIO(img_bytes), include_gradcam=True)

            # Ordinary prediction must remain intact
            self.assertIsNone(res['error'])
            self.assertIn(res['predicted_class'], CLASS_INDEX_MAP)
            self.assertGreater(res['predicted_prob'], 0.0)
            self.assertEqual(len(res['all_probabilities']), 7)

            # Grad-CAM reflects failure gracefully
            self.assertIn('gradcam', res)
            self.assertFalse(res['gradcam']['success'])
            self.assertIsNotNone(res['gradcam']['error'])
        finally:
            inf._conv_extractor = original_extractor

    def test_gradcam_invalid_input_graceful_handling(self):
        """Confirm invalid/corrupted bytes passed to generate_gradcam return graceful error."""
        result = generate_gradcam(b'corrupted_raw_bytes', target_class_index=0)
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIsNone(result['heatmap_data_url'])
        self.assertIsNone(result['overlay_data_url'])


# ==============================================================================
# 3. Image Upload Validation Tests
# ==============================================================================

class ImageUploadValidationTests(TestCase):
    """Verifies upload validation for valid formats, extensions, sizes, and malicious/corrupt inputs."""

    def test_valid_jpg_upload(self):
        """Test valid JPG image upload passes validation."""
        _, data = _create_synthetic_lesion_image(format='JPEG')
        f = SimpleUploadedFile('sample.jpg', data, content_type='image/jpeg')
        form = ImageUploadForm({}, {'image': f})
        self.assertTrue(form.is_valid())

    def test_valid_jpeg_upload(self):
        """Test valid JPEG image upload passes validation."""
        _, data = _create_synthetic_lesion_image(format='JPEG')
        f = SimpleUploadedFile('sample.jpeg', data, content_type='image/jpeg')
        form = ImageUploadForm({}, {'image': f})
        self.assertTrue(form.is_valid())

    def test_valid_png_upload(self):
        """Test valid PNG image upload passes validation."""
        _, data = _create_synthetic_lesion_image(format='PNG')
        f = SimpleUploadedFile('sample.png', data, content_type='image/png')
        form = ImageUploadForm({}, {'image': f})
        self.assertTrue(form.is_valid())

    def test_empty_post_validation_error(self):
        """Test submitting form with no file triggers required validation error."""
        form = ImageUploadForm({}, {})
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)
        self.assertIn('Please select an image file.', str(form.errors['image']))

    def test_zero_byte_file_upload(self):
        """Test 0-byte file triggers empty file validation error."""
        f = SimpleUploadedFile('empty.jpg', b'', content_type='image/jpeg')
        form = ImageUploadForm({}, {'image': f})
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)

    def test_invalid_file_extension(self):
        """Test disallowed file extensions (.txt, .pdf, .gif, .exe) are rejected."""
        disallowed = [
            ('document.pdf', b'%PDF-1.4...', 'application/pdf'),
            ('notes.txt', b'hello world', 'text/plain'),
            ('animation.gif', b'GIF89a...', 'image/gif'),
            ('script.py', b'print(1)', 'text/x-python'),
        ]

        for fname, content, ctype in disallowed:
            with self.subTest(filename=fname):
                f = SimpleUploadedFile(fname, content, content_type=ctype)
                form = ImageUploadForm({}, {'image': f})
                self.assertFalse(form.is_valid())
                self.assertIn('image', form.errors)

    def test_corrupted_image_file(self):
        """Test non-image file with a .jpg extension is rejected by Pillow image verification."""
        f = SimpleUploadedFile('corrupt.jpg', b'THIS_IS_NOT_A_VALID_IMAGE_DATA_12345', content_type='image/jpeg')
        form = ImageUploadForm({}, {'image': f})
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)

    def test_oversized_file_rejected(self):
        """Test valid image exceeding MAX_UPLOAD_SIZE limit (10MB) is rejected."""
        # Create uncompressed large image > 10MB
        large_img = Image.new('RGB', (2500, 2000), color=(180, 140, 130))
        buf = io.BytesIO()
        large_img.save(buf, format='PNG', compress_level=0)
        data = buf.getvalue()
        self.assertGreater(len(data), 10 * 1024 * 1024)

        f = SimpleUploadedFile('too_large.png', data, content_type='image/png')
        form = ImageUploadForm({}, {'image': f})
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)
        self.assertIn('File size exceeds', str(form.errors['image']))


# ==============================================================================
# 4. Django Functionality & View Tests
# ==============================================================================

class DjangoViewAndPageTests(TestCase):
    """Verifies web views, navigation, CSRF protection, and zero disk-storage policy."""

    def setUp(self):
        self.client = Client()

    def test_home_page_renders_successfully(self):
        """Test GET / returns HTTP 200, contains model status, statistics, and medical disclaimer."""
        response = self.client.get(reverse('classifier:home'))
        self.assertEqual(response.status_code, 200)

        content = response.content.decode('utf-8')
        self.assertIn('Skin Disease', content)
        self.assertIn('Classification', content)
        self.assertIn('Medical Disclaimer', content)
        self.assertIn('not a medical diagnosis', content)
        self.assertIn('MobileNetV2', content)

    def test_analyze_page_get_renders_form_and_csrf(self):
        """Test GET /analyze/ returns HTTP 200 with upload form, dropzone, and CSRF token."""
        response = self.client.get(reverse('classifier:analyze'))
        self.assertEqual(response.status_code, 200)

        content = response.content.decode('utf-8')
        self.assertIn('Upload Image', content)
        self.assertIn('dropzone', content)
        self.assertIn('csrfmiddlewaretoken', content)
        self.assertIn('Analyze Image', content)

    def test_results_page_without_session_redirects_to_analyze(self):
        """Test directly accessing GET /results/ without a session redirects to /analyze/."""
        response = self.client.get(reverse('classifier:results'), follow=True)
        self.assertEqual(response.status_code, 200)

        # Redirect chain should point to analyze
        self.assertTrue(any('analyze' in str(url) for url, _ in response.redirect_chain))

    def test_end_to_end_analyze_to_results_workflow(self):
        """Test end-to-end: POST valid image -> redirect to /results/ -> verify all UI elements."""
        w, h = 600, 450
        _, data = _create_synthetic_lesion_image(width=w, height=h, format='JPEG')
        uploaded_file = SimpleUploadedFile('test_lesion.jpg', data, content_type='image/jpeg')

        response = self.client.post(
            reverse('classifier:analyze'),
            {'image': uploaded_file},
            follow=True
        )

        self.assertEqual(response.status_code, 200)
        content = response.content.decode('utf-8')

        # 1. Prediction summary & probability
        self.assertIn('Analysis Results', content)
        self.assertIn('Top Prediction', content)
        self.assertIn('All Class Probabilities', content)
        self.assertIn('confidence', content)

        # 2. Grad-CAM section and 3 panels
        self.assertIn('Grad-CAM Visual Explanations', content)
        self.assertIn('Original Image', content)
        self.assertIn('Grad-CAM Heatmap', content)
        self.assertIn('Heatmap Overlay', content)

        # 3. Correct dimensions displayed
        self.assertIn(f'{w} × {h} px', content)

        # 4. Interactive transparency controls
        self.assertIn('id="transparency-slider"', content)
        self.assertIn('id="opacity-value-badge"', content)
        self.assertIn('data-value="50"', content)

        # 5. Educational and medical disclaimer (Requirement 7)
        self.assertIn('Understanding the Visual Explanation', content)
        self.assertIn('not a definitive explanation or confirmation of a medical condition', content)
        self.assertIn('educational prototype', content)

    def test_no_permanent_files_saved_to_disk(self):
        """Verify that running image prediction does not create any files inside media directory."""
        media_root = getattr(settings, 'MEDIA_ROOT', None)
        before_files = set()
        if media_root and os.path.exists(media_root):
            before_files = set(os.listdir(media_root))

        _, data = _create_synthetic_lesion_image(width=400, height=300)
        f = SimpleUploadedFile('disk_leak_test.jpg', data, content_type='image/jpeg')
        self.client.post(reverse('classifier:analyze'), {'image': f}, follow=True)

        after_files = set()
        if media_root and os.path.exists(media_root):
            after_files = set(os.listdir(media_root))

        self.assertEqual(before_files, after_files, "New files were detected on disk in MEDIA_ROOT!")

    def test_logger_safety(self):
        """Verify inference logging contains only classification status and no sensitive data."""
        from classifier.inference import logger
        with self.assertLogs(logger, level='INFO') as log:
            _, data = _create_synthetic_lesion_image(width=128, height=128)
            predict(io.BytesIO(data), include_gradcam=False)

            log_output = "\n".join(log.output)
            # Must log prediction class and prob
            self.assertIn("Prediction: class=", log_output)
            # Must not log binary or base64 raw payloads
            self.assertNotIn("data:image", log_output)
            self.assertNotIn("b'\\x", log_output)
