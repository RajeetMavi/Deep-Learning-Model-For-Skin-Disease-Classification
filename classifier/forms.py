"""
Django forms for the classifier app.
Handles image upload validation.
"""

import os
from django import forms
from django.conf import settings


class ImageUploadForm(forms.Form):
    """
    Form for uploading a skin lesion image for classification.
    Validates file type, extension, and size.
    """
    image = forms.ImageField(
        label='Upload Skin Lesion Image',
        help_text='Accepted formats: JPG, JPEG, PNG. Maximum size: 10 MB.',
        error_messages={
            'required': 'Please select an image file.',
            'invalid_image': 'The uploaded file is not a valid image.',
        }
    )

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('Please select an image file.')

        # Check file extension
        ext = os.path.splitext(image.name)[1].lower()
        allowed_exts = getattr(settings, 'ALLOWED_IMAGE_EXTENSIONS', ['.jpg', '.jpeg', '.png'])
        if ext not in allowed_exts:
            raise forms.ValidationError(
                f'Unsupported file type "{ext}". Please upload a JPG, JPEG, or PNG image.'
            )

        # Check content type
        content_type = getattr(image, 'content_type', '')
        allowed_types = getattr(settings, 'ALLOWED_IMAGE_TYPES', ['image/jpeg', 'image/jpg', 'image/png'])
        if content_type and content_type not in allowed_types:
            raise forms.ValidationError(
                'Unsupported image format. Please upload a JPG or PNG image.'
            )

        # Check file size
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 10 * 1024 * 1024)
        if image.size > max_size:
            size_mb = max_size / (1024 * 1024)
            raise forms.ValidationError(
                f'File size exceeds the {size_mb:.0f} MB limit. Please upload a smaller image.'
            )

        return image
