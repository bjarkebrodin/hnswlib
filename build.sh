#!/usr/bin/env zsh
cd build && make all && \
cd - && python setup.py build
