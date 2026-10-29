/**
 * upload.js — Image Upload Page JavaScript
 * Handles: drag-and-drop, file selection, preview, client-side validation,
 *          loading state on form submit.
 */

(function () {
    'use strict';

    const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
    const ALLOWED_TYPES  = ['image/jpeg', 'image/jpg', 'image/png'];
    const ALLOWED_EXTS   = ['.jpg', '.jpeg', '.png'];

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const dropzone       = document.getElementById('dropzone');
    const fileInput      = document.getElementById('id_image');  // Django renders with this ID
    const browseBtn      = document.getElementById('browse-btn');
    const clearBtn       = document.getElementById('clear-btn');
    const analyzeBtn     = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const analyzeBtnLoad = document.getElementById('analyze-btn-loading');
    const dropzoneIdle   = document.getElementById('dropzone-idle');
    const dropzonePreview= document.getElementById('dropzone-preview');
    const previewImg     = document.getElementById('preview-img');
    const previewFilename= document.getElementById('preview-filename');
    const previewSize    = document.getElementById('preview-size');
    const clientError    = document.getElementById('client-error');
    const clientErrorMsg = document.getElementById('client-error-msg');
    const uploadForm     = document.getElementById('upload-form');

    if (!dropzone || !fileInput) return; // guard: not on upload page

    // ── Helpers ───────────────────────────────────────────────────────────────
    function formatBytes(bytes) {
        if (bytes < 1024)       return bytes + ' B';
        if (bytes < 1048576)    return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(2) + ' MB';
    }

    function getExt(filename) {
        return filename.slice(filename.lastIndexOf('.')).toLowerCase();
    }

    function showError(msg) {
        if (clientError && clientErrorMsg) {
            clientErrorMsg.textContent = '⚠️ ' + msg;
            clientError.style.display = 'block';
        }
    }

    function clearError() {
        if (clientError) clientError.style.display = 'none';
    }

    function validateFile(file) {
        if (!file) return 'No file selected.';
        const ext = getExt(file.name);
        if (!ALLOWED_EXTS.includes(ext)) {
            return `Unsupported file type "${ext}". Please upload a JPG, JPEG, or PNG image.`;
        }
        if (file.type && !ALLOWED_TYPES.includes(file.type)) {
            return 'Unsupported MIME type. Please upload a JPG or PNG image.';
        }
        if (file.size > MAX_SIZE_BYTES) {
            return `File is too large (${formatBytes(file.size)}). Maximum size is 10 MB.`;
        }
        return null; // valid
    }

    // ── Preview ───────────────────────────────────────────────────────────────
    function showPreview(file) {
        const error = validateFile(file);
        if (error) {
            showError(error);
            resetPreview();
            return;
        }
        clearError();

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewFilename.textContent = file.name;
            previewSize.textContent = formatBytes(file.size);
            dropzoneIdle.style.display    = 'none';
            dropzonePreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetPreview() {
        previewImg.src = '';
        previewFilename.textContent = '';
        previewSize.textContent = '';
        dropzoneIdle.style.display    = 'flex';
        dropzonePreview.style.display = 'none';
        analyzeBtn.disabled = true;
        fileInput.value = '';
    }

    // ── Browse button ─────────────────────────────────────────────────────────
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Dropzone also opens file picker on click/Enter
    dropzone.addEventListener('click', (e) => {
        if (e.target === clearBtn || clearBtn?.contains(e.target)) return;
        if (e.target === browseBtn || browseBtn?.contains(e.target)) return;
        // Only open picker if in idle state
        if (dropzonePreview.style.display === 'none' || !dropzonePreview.style.display) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('keydown', (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && dropzonePreview.style.display !== 'flex') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // ── File input change ─────────────────────────────────────────────────────
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) showPreview(file);
    });

    // ── Clear / Remove button ─────────────────────────────────────────────────
    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetPreview();
            clearError();
        });
    }

    // ── Drag and drop ─────────────────────────────────────────────────────────
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', (e) => {
        if (!dropzone.contains(e.relatedTarget)) {
            dropzone.classList.remove('drag-over');
        }
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const file = e.dataTransfer?.files[0];
        if (file) {
            // Transfer the file to the hidden input via DataTransfer API
            try {
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
            } catch (_) {
                // DataTransfer not supported — just show preview (won't submit)
            }
            showPreview(file);
        }
    });

    // ── Form submit: show loading state ──────────────────────────────────────
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            // Re-validate before submit
            const file = fileInput.files[0];
            const error = validateFile(file);
            if (error) {
                e.preventDefault();
                showError(error);
                return;
            }

            // Show loading state
            if (analyzeBtnText) analyzeBtnText.style.display = 'none';
            if (analyzeBtnLoad) analyzeBtnLoad.style.display = 'inline-flex';
            analyzeBtn.disabled = true;
            analyzeBtn.setAttribute('aria-busy', 'true');
        });
    }

})();
