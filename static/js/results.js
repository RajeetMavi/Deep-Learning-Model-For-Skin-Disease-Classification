/**
 * results.js — Results Page JavaScript
 * Handles:
 *   - Animated probability bars (triggered on page load)
 *   - Circular probability gauge animation
 *   - SVG gradient injection for the gauge
 */

(function () {
    'use strict';

    // ── SVG Gradient for the circular gauge ───────────────────────────────────
    // Inject a <defs> block with a linearGradient at the top of <body>
    const svgDefs = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgDefs.setAttribute('width', '0');
    svgDefs.setAttribute('height', '0');
    svgDefs.setAttribute('style', 'position:absolute;');
    svgDefs.setAttribute('aria-hidden', 'true');
    svgDefs.innerHTML = `
        <defs>
            <linearGradient id="prob-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%"   stop-color="hsl(220,90%,60%)"/>
                <stop offset="100%" stop-color="hsl(280,80%,65%)"/>
            </linearGradient>
        </defs>
    `;
    document.body.insertBefore(svgDefs, document.body.firstChild);

    // ── Animate circular gauge ─────────────────────────────────────────────────
    const circleFill = document.querySelector('.prob-circle-fill');
    if (circleFill) {
        const prob = parseFloat(circleFill.getAttribute('data-prob') || '0');
        const circumference = 314; // 2 * π * r ≈ 2 * 3.14159 * 50
        const offset = circumference - (prob / 100) * circumference;

        // Start from 0 (full offset = no fill)
        circleFill.style.strokeDashoffset = circumference.toString();

        // Animate after a short delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                circleFill.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.22, 1, 0.36, 1)';
                circleFill.style.strokeDashoffset = offset.toFixed(2);
            }, 200);
        });
    }

    // ── Animate probability bars ──────────────────────────────────────────────
    const bars = document.querySelectorAll('.prob-bar-fill[data-width]');

    if (bars.length > 0) {
        // Use IntersectionObserver so bars animate when they scroll into view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const bar = entry.target;
                    const targetWidth = bar.getAttribute('data-width');
                    // Small stagger based on index within the list
                    const index = Array.from(bars).indexOf(bar);
                    setTimeout(() => {
                        bar.style.width = targetWidth + '%';
                    }, 100 + index * 80);
                    observer.unobserve(bar);
                }
            });
        }, { threshold: 0.1 });

        bars.forEach(bar => observer.observe(bar));
    }

    // ── Animate prob-item cards stagger ───────────────────────────────────────
    const probItems = document.querySelectorAll('.prob-item');
    probItems.forEach((item, i) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(12px)';
        item.style.transition = `opacity 0.4s ease, transform 0.4s ease`;
        setTimeout(() => {
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, 150 + i * 60);
    });

    // ── Animate the top prediction card ──────────────────────────────────────
    const topCard = document.querySelector('.top-prediction-card');
    if (topCard) {
        topCard.style.opacity = '0';
        topCard.style.transform = 'translateY(16px)';
        topCard.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        setTimeout(() => {
            topCard.style.opacity = '1';
            topCard.style.transform = 'translateY(0)';
        }, 100);
    }

    // ── Grad-CAM Overlay Transparency Slider ──────────────────────────────────
    const transparencySlider = document.getElementById('transparency-slider');
    const heatmapLayer = document.getElementById('gradcam-heatmap-layer');
    const opacityBadge = document.getElementById('opacity-value-badge');
    const presetButtons = document.querySelectorAll('.transparency-presets .preset-btn');

    if (transparencySlider && heatmapLayer && opacityBadge) {
        function setTransparency(val) {
            const num = Math.max(0, Math.min(100, parseInt(val, 10) || 0));
            transparencySlider.value = num;
            heatmapLayer.style.opacity = (num / 100).toFixed(2);
            opacityBadge.textContent = `${num}%`;

            presetButtons.forEach(btn => {
                const btnVal = parseInt(btn.getAttribute('data-value'), 10);
                if (btnVal === num) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        transparencySlider.addEventListener('input', (e) => {
            setTransparency(e.target.value);
        });

        presetButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const val = btn.getAttribute('data-value');
                setTransparency(val);
            });
        });
    }

    // ── Animate Grad-CAM gallery cards entrance ───────────────────────────────
    const gradcamCards = document.querySelectorAll('.gradcam-card');
    gradcamCards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(14px)';
        card.style.transition = 'opacity 0.45s ease, transform 0.45s ease';
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 300 + i * 80);
    });

})();
