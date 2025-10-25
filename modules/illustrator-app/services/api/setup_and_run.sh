#!/bin/bash

uv run setup.py

uv run manage.py rundramatiq -p 2 -t 1 &
DRAMATIQ_PID=$!

uv run manage.py runserver 0.0.0.0:8000 &
RUNSERVER_PID=$!

cleanup() {
    echo "Shutting down processes..."
    kill $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
    wait $DRAMATIQ_PID $RUNSERVER_PID 2>/dev/null
}

trap cleanup EXIT INT TERM

wait $DRAMATIQ_PID $RUNSERVER_PID
