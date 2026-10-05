#!/bin/bash
# Export every KEY=VALUE in the repo .env that is not already set.
#
# Sourced, not executed. The environment wins over .env, as in data/paths.py.
# Values are read literally (quotes stripped, no expansion); within .env the
# last assignment wins.

if [[ -z "${_DOTENV_LOADED:-}" ]]; then
    _dotenv_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
    _dotenv_ours=""
    if [[ -f "$_dotenv_file" ]]; then
        while IFS= read -r _dotenv_line || [[ -n "$_dotenv_line" ]]; do
            [[ "$_dotenv_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
            _dotenv_key="${BASH_REMATCH[1]}"
            _dotenv_val="${BASH_REMATCH[2]}"
            if [[ "$_dotenv_val" =~ ^\"([^\"]*)\" || "$_dotenv_val" =~ ^\'([^\']*)\' ]]; then
                _dotenv_val="${BASH_REMATCH[1]}"
            else
                _dotenv_val="${_dotenv_val%%#*}"
                _dotenv_val="${_dotenv_val%"${_dotenv_val##*[![:space:]]}"}"
            fi
            if [[ -z "${!_dotenv_key+set}" || " $_dotenv_ours " == *" $_dotenv_key "* ]]; then
                export "$_dotenv_key=$_dotenv_val"
                _dotenv_ours+=" $_dotenv_key"
            fi
        done < "$_dotenv_file"
    fi
    export _DOTENV_LOADED=1
    unset _dotenv_file _dotenv_line _dotenv_key _dotenv_val _dotenv_ours
fi
