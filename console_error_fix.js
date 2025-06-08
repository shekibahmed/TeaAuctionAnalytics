// Console error suppression for known harmless warnings
(function() {
    'use strict';
    
    // Store original console methods
    const originalError = console.error;
    const originalWarn = console.warn;
    
    // List of known harmless error patterns to suppress
    const suppressPatterns = [
        /ResizeObserver loop completed with undelivered notifications/,
        /ResizeObserver loop limit exceeded/,
        /Non-passive event listener/,
        /Gather usage stats/
    ];
    
    // Override console.error to filter out harmless warnings
    console.error = function(...args) {
        const message = args.join(' ');
        
        // Check if this is a harmless error we should suppress
        const shouldSuppress = suppressPatterns.some(pattern => 
            pattern.test(message)
        );
        
        if (!shouldSuppress) {
            originalError.apply(console, args);
        }
    };
    
    // Override console.warn for similar filtering
    console.warn = function(...args) {
        const message = args.join(' ');
        
        const shouldSuppress = suppressPatterns.some(pattern => 
            pattern.test(message)
        );
        
        if (!shouldSuppress) {
            originalWarn.apply(console, args);
        }
    };
    
    // Add ResizeObserver error handling
    window.addEventListener('error', function(event) {
        if (event.message && event.message.includes('ResizeObserver')) {
            event.preventDefault();
            return false;
        }
    });
    
})();