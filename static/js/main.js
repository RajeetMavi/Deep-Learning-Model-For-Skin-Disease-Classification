/**
 * main.js — Global site JavaScript
 * Handles: navbar scroll effects, mobile menu, alert dismissal
 */

(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Navbar: add .scrolled class when user scrolls down
    // -------------------------------------------------------------------------
    const navbar = document.getElementById('main-nav');
    if (navbar) {
        const onScroll = () => {
            navbar.classList.toggle('scrolled', window.scrollY > 20);
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll(); // run once on load
    }

    // -------------------------------------------------------------------------
    // Mobile navigation toggle
    // -------------------------------------------------------------------------
    const navToggle = document.getElementById('nav-toggle');
    const navMenu   = document.getElementById('nav-menu');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            const isOpen = navMenu.classList.toggle('open');
            navToggle.setAttribute('aria-expanded', isOpen.toString());
        });

        // Close menu when a link is clicked
        navMenu.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('open');
                navToggle.setAttribute('aria-expanded', 'false');
            });
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
                navMenu.classList.remove('open');
                navToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // -------------------------------------------------------------------------
    // Alert / notification dismissal
    // -------------------------------------------------------------------------
    document.querySelectorAll('.alert-close').forEach(btn => {
        btn.addEventListener('click', () => {
            const alert = btn.closest('.alert');
            if (alert) {
                alert.style.transition = 'opacity 200ms, transform 200ms';
                alert.style.opacity = '0';
                alert.style.transform = 'translateX(20px)';
                setTimeout(() => alert.remove(), 220);
            }
        });
    });

    // Auto-dismiss alerts after 6 seconds
    document.querySelectorAll('.alert').forEach((alert, i) => {
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.transition = 'opacity 400ms, transform 400ms';
                alert.style.opacity = '0';
                alert.style.transform = 'translateX(20px)';
                setTimeout(() => alert.remove(), 420);
            }
        }, 6000 + i * 300);
    });

    // -------------------------------------------------------------------------
    // Smooth scroll for anchor links
    // -------------------------------------------------------------------------
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

})();
