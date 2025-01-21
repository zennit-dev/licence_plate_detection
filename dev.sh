#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_DIR="${SCRIPT_DIR}/docker"
export DOCKER_COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"

# Default environment
export BUILD_ENV=${BUILD_ENV:-dev}

# Change to the project root directory
cd "${SCRIPT_DIR}" || {
    echo -e "${RED}Failed to change to project directory: ${SCRIPT_DIR}${NC}"
    exit 1
}

show_help() {
    echo -e "\n${BLUE}ML Development Environment Commands:${NC}"
    echo -e "\nUsage: $0 {help|start|up|down|clean|fresh} [env]"
    echo -e "\nAvailable commands:"
    echo -e "  ${GREEN}help${NC}    Show this help message"
    echo -e "  ${GREEN}start${NC}   Build containers, start environment, and enter shell"
    echo -e "  ${GREEN}up${NC}      Start environment and enter shell (no rebuild)"
    echo -e "  ${GREEN}down${NC}    Stop development environment"
    echo -e "  ${GREEN}clean${NC}   Stop and remove containers, images, and volumes"
    echo -e "  ${GREEN}fresh${NC}   Clean and start fresh environment"
    echo -e "\nEnvironment options:"
    echo -e "  ${GREEN}dev${NC}     Development environment (default)"
    echo -e "  ${GREEN}prod${NC}    Production environment"
    echo
}

start_environment() {
    local should_build=$1
    local container_name="ml-tensorflow-${BUILD_ENV}"

    if [ "$should_build" = true ]; then
        echo -e "${BLUE}Building Docker containers (${BUILD_ENV} environment)...${NC}"
        cd "${DOCKER_DIR}" && docker compose --profile "${BUILD_ENV}" build
    fi
    
    echo -e "${BLUE}Starting environment (${BUILD_ENV})...${NC}"
    cd "${DOCKER_DIR}" && docker compose --profile "${BUILD_ENV}" up -d
    
    echo -e "${BLUE}Entering container...${NC}"
    docker exec -it "${container_name}" bash || {
        echo -e "${RED}Failed to enter container${NC}"
        exit 1
    }
}

stop_environment() {
    echo -e "${BLUE}Stopping environment...${NC}"
    cd "${DOCKER_DIR}" && docker compose --profile "${BUILD_ENV}" down
}

clean_environment() {
    echo -e "${BLUE}Cleaning up environment...${NC}"
    
    # Check if there are any containers from our compose file
    if [ -n "$(cd "${DOCKER_DIR}" && docker compose --profile "${BUILD_ENV}" ps -q)" ]; then
        cd "${DOCKER_DIR}" && docker compose --profile "${BUILD_ENV}" down --rmi all --volumes --remove-orphans
    else
        echo -e "${BLUE}Nothing to clean up${NC}"
    fi
}

start_fresh() {
    echo -e "${BLUE}Starting fresh environment...${NC}"
    clean_environment
    start_environment true
}

# Check for environment argument
if [ "$2" = "prod" ]; then
    export BUILD_ENV=prod
elif [ "$2" = "dev" ]; then
    export BUILD_ENV=dev
elif [ -n "$2" ]; then
    echo -e "${RED}Invalid environment: $2${NC}"
    show_help
    exit 1
fi

case "$1" in
    "help"|"")
        show_help
        ;;
    "start")
        start_environment true
        ;;
    "up")
        start_environment false
        ;;
    "down")
        stop_environment
        ;;
    "clean")
        clean_environment
        ;;
    "fresh")
        start_fresh
        ;;
    *)
        echo -e "${RED}Invalid command: $1${NC}"
        show_help
        exit 1
        ;;
esac 