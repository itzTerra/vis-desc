#!/bin/bash

uv run manage.py rundramatiq -p 1 -t 1 &
DRAMATIQ_PID=$!

# uv run daphne -b 0.0.0.0 -p "$PORT" api.asgi:application &
uv run manage.py runserver 0.0.0.0:$PORT &
RUNSERVER_PID=$!

cleanup() {
    echo "Shutting down processes..."
    kill $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
    wait $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
}

trap cleanup EXIT INT TERM

wait $DRAMATIQ_PID $RUNSERVER_PID
