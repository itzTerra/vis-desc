#!/bin/sh
if [ ! -d node_modules ] || [ package.json -nt node_modules ]; then
  npm install --include=dev .

  if [ "$(uname)" = "Darwin" ]; then
    [ -d node_modules ] && touch node_modules
  else
    touch --no-create node_modules
  fi
fi

