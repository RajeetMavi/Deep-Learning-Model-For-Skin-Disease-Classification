"""
Views for the Skin Lesion Classification app.

Handles:
  - Home page
  - Image upload / analysis page  (POST → runs real inference)
  - Results page
"""

import io
import base64
import logging
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.contrib import messages

from .forms import ImageUploadForm
from .inference import predict, get_model_status, CLASS_INFO

logger = logging.getLogger(__name__)


def home(request):
    """Landing page with project overview and category descriptions."""
    model_loaded, model_error = get_model_status()
    context = {
        'class_info': CLASS_INFO,
        'model_loaded': model_loaded,
        'model_error': model_error,
        'page_title': 'Skin Lesion Classification — AI-Powered Dermatology Tool',
    }
    return render(request, 'classifier/home.html', context)


@require_http_methods(["GET", "POST"])
def analyze(request):
    """
    Image upload page.
    GET  → show empty upload form.
    POST → validate, preprocess, run real model inference, redirect to results.
    """
    model_loaded, model_error = get_model_status()

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']

            # --- Encode image as base64 for session storage (no disk writes) ---
            image_file.seek(0)
            image_bytes = image_file.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            name = (getattr(image_file, 'name', '') or '').lower()
            mime = 'image/png' if name.endswith('.png') else 'image/jpeg'
            image_data_url = f'data:{mime};base64,{image_b64}'

            # --- Run real model inference ---
            image_file.seek(0)
            result = predict(image_file)

            if result.get('error'):
                # Surface the error to the user without a crash
                messages.error(
                    request,
                    f"⚠️ Prediction failed: {result['error']}"
                )
                return render(request, 'classifier/analyze.html', {
                    'form': form,
                    'model_loaded': model_loaded,
                    'model_error': model_error,
                    'page_title': 'Analyze Image — Skin Lesion Classification',
                })

            # --- Persist result + image in session ---
            request.session['prediction_result'] = result
            request.session['uploaded_image'] = image_data_url
            request.session.modified = True

            logger.info(
                "Prediction OK  class=%s  prob=%.4f  is_mock=%s",
                result.get('predicted_class'),
                result.get('predicted_prob', 0),
                result.get('is_mock', False),
            )

            return redirect('classifier:results')

        else:
            # Django form validation failed — re-render with error messages
            return render(request, 'classifier/analyze.html', {
                'form': form,
                'model_loaded': model_loaded,
                'model_error': model_error,
                'page_title': 'Analyze Image — Skin Lesion Classification',
            })

    # GET — show fresh form
    form = ImageUploadForm()
    return render(request, 'classifier/analyze.html', {
        'form': form,
        'model_loaded': model_loaded,
        'model_error': model_error,
        'page_title': 'Analyze Image — Skin Lesion Classification',
    })


def results(request):
    """Results page — reads prediction from session, renders probability breakdown."""
    result = request.session.get('prediction_result')
    image_data_url = request.session.get('uploaded_image')

    if not result:
        messages.info(request, 'Please upload an image to get a prediction.')
        return redirect('classifier:analyze')

    risk_colors = {
        'low':      'success',
        'medium':   'warning',
        'high':     'danger',
        'critical': 'critical',
    }

    gradcam = result.get('gradcam') or {}

    context = {
        'result':          result,
        'image_data_url':  image_data_url,
        'gradcam':         gradcam,
        'risk_color':      risk_colors.get(result.get('predicted_risk', 'low'), 'secondary'),
        'page_title':      f"Results: {result.get('predicted_short', 'Analysis')} — Skin Lesion Classification",
    }
    return render(request, 'classifier/results.html', context)
