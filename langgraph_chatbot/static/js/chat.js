/**
 * static/js/chat.js
 * Handles cooking loader animation cycling
 */

(function() {
    'use strict';
    
    // Animation frames matching Claude's working code
    const FRAMES = [
        "👩🏻‍🍳 Bro's cooking. Let him cook 🔥",
        "👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥",
        "👩🏻‍🍳👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥🔥"
    ];
    
    let animationInterval = null;
    let currentFrame = 0;
    
    /**
     * Updates the cooking loader text
     */
    function updateLoaderText() {
        const loaderDiv = document.getElementById('cooking-loader');
        if (!loaderDiv) return;
        
        currentFrame = (currentFrame + 1) % FRAMES.length;
        
        // Update the content
        loaderDiv.innerHTML = `
            <span class="cooking-emoji">${FRAMES[currentFrame].split(' ')[0]}</span>
            <span class="cooking-text">${FRAMES[currentFrame].split(' ').slice(1).join(' ')}</span>
        `;
    }
    
    /**
     * Starts the cooking loader animation
     */
    function startCookingAnimation() {
        stopCookingAnimation();
        
        const loaderDiv = document.getElementById('cooking-loader');
        if (loaderDiv) {
            loaderDiv.style.display = 'flex';
            animationInterval = setInterval(updateLoaderText, 500);
        }
    }
    
    /**
     * Stops the cooking loader animation
     */
    function stopCookingAnimation() {
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        
        const loaderDiv = document.getElementById('cooking-loader');
        if (loaderDiv) {
            loaderDiv.style.display = 'none';
        }
    }
    
    // Export functions to global scope
    window.chat = {
        startCookingAnimation: startCookingAnimation,
        stopCookingAnimation: stopCookingAnimation
    };
})();