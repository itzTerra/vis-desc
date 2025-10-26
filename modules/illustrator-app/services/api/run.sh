#!/bin/bash

if [ "$ENABLE_DRAMATIQ" == "off" ]; then
    echo "Dramatiq is disabled. Running only the API server."
    uv run daphne -b 0.0.0.0 -p $PORT api.asgi:application
    exit 0
fi

uv run daphne -b 0.0.0.0 -p $PORT api.asgi:application &
# uv run manage.py runserver 0.0.0.0:$PORT &
RUNSERVER_PID=$!
echo "API server started with PID $RUNSERVER_PID"

uv run manage.py rundramatiq -p 1 -t 1 &
DRAMATIQ_PID=$!
echo "Dramatiq worker started with PID $DRAMATIQ_PID"

cleanup() {
    echo "Shutting down processes..."
    kill $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
    wait $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
    # kill $RUNSERVER_PID 2>/dev/null
    # wait $RUNSERVER_PID 2>/dev/null
    echo "Processes terminated."
}

trap cleanup EXIT INT TERM

wait $DRAMATIQ_PID $RUNSERVER_PID
# wait $RUNSERVER_PID
