#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [-c] [-h]"
    echo "  -c    Clean screenshots and reset database before starting"
    echo "  -h    Display this help message"
}

# Function to clean screenshots and reset database
clean_data() {
    echo "Cleaning screenshots and resetting database..."
    rm -f static/screenshots/*
    rm -f screenshots.db
    echo "Cleanup complete."
}

# Function to stop all services
stop_services() {
    echo "Stopping services..."
    
    # Stop Flask
    if [ ! -z "$FLASK_PID" ]; then
        kill -TERM "$FLASK_PID" 2>/dev/null
        wait "$FLASK_PID" 2>/dev/null
    fi

    # Stop Celery
    if [ ! -z "$CELERY_PID" ]; then
        kill -TERM "$CELERY_PID" 2>/dev/null
        wait "$CELERY_PID" 2>/dev/null
    fi

    # Stop Redis container
    docker stop redis-server
    docker rm redis-server

    echo "All services stopped."
    exit 0
}

# Set up trap to call stop_services on script exit
trap stop_services EXIT INT TERM

# Function to ensure NLTK data is downloaded
ensure_nltk_data() {
    echo "Ensuring NLTK data is available..."
    python - <<EOF
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data check complete.")
EOF
}

# Parse command line options
while getopts ":ch" opt; do
    case ${opt} in
        c )
            clean_data
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            show_help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Ensure NLTK data is downloaded
ensure_nltk_data

# Start Redis using Docker
echo "Starting Redis using Docker..."
docker run --name redis-server -p 6379:6379 -d redis

# Wait for Redis to start
sleep 2

# Check if Redis container is running
if ! docker ps | grep -q redis-server; then
    echo "Failed to start Redis container. Please check Docker and Redis image and try again."
    exit 1
fi

echo "Redis server started successfully."

# Start Celery worker
echo "Starting Celery worker..."
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
celery -A main.celery worker --loglevel=info --concurrency=2 &
CELERY_PID=$!

# Wait for Celery to start
sleep 5

# Check if Celery is running
if ! kill -0 $CELERY_PID 2>/dev/null; then
    echo "Failed to start Celery worker. Please check Celery installation and try again."
    exit 1
fi

echo "Celery worker started successfully."

# Start Flask application
echo "Starting Flask application..."
python main.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 2

# Check if Flask is running
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "Failed to start Flask application. Please check Flask installation and try again."
    exit 1
fi

echo "Flask application started successfully."

echo "All services started. Press Ctrl+C to stop."

# Wait for user interrupt
wait