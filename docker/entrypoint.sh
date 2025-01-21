#!/bin/bash
set -e

# Initialize any services or environment if needed
initialize() {
    echo "Initializing development environment..."
    
    # Install dependencies if needed
    if [ ! -d "/app/.venv" ] || [ -z "$(ls -A /app/.venv)" ]; then
        echo "Installing dependencies..."
        poetry install
    fi

    # Add .venv/bin to PATH if not already there
    if [[ ":$PATH:" != *":/app/.venv/bin:"* ]]; then
        export PATH="/app/.venv/bin:$PATH"
    fi
}

# Main container logic
main() {
    # Always initialize first
    initialize
    
    echo "Development environment is ready!"
    
    # Execute the command if provided
    if [ $# -gt 0 ]; then
        exec "$@"
    else
        # If no command provided, just sleep infinity
        # This allows the container to stay running for docker exec
        exec tail -f /dev/null
    fi
}

# Execute main function with all arguments
main "$@" 