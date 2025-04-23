#!/bin/bash

# --- Configuration ---
# Set the path to the file containing your Hugging Face token
TOKEN_FILE="/pvc/hf_token.txt" # <--- IMPORTANT: Replace with the actual path

# --- Script Logic ---

# 1. Check if the token file exists
if [ ! -f "$TOKEN_FILE" ]; then
  echo "Error: Token file not found at '$TOKEN_FILE'" >&2 # Print error to stderr
  exit 1 # Exit with a non-zero status indicating failure
fi

# 2. Read the token from the file
# Use 'cat' and command substitution $() to capture the file content.
HF_TOKEN=$(cat "$TOKEN_FILE")

# 3. Check if the token was read successfully (is not empty)
if [ -z "$HF_TOKEN" ]; then
  echo "Error: Token file '$TOKEN_FILE' appears to be empty." >&2
  exit 1
fi

# 4. Log in using the token non-interactively, without affecting git
echo "Attempting to log in to Hugging Face CLI using token..."
# Provide the token directly via --token
# Crucially, DO NOT include --add-to-git-credential
huggingface-cli login --token "$HF_TOKEN"

# 5. Check the exit status of the login command
LOGIN_STATUS=$? # Capture the exit code of the last command
if [ $LOGIN_STATUS -eq 0 ]; then
  echo "Hugging Face CLI login successful."
  echo "NOTE: Token was used for login but *not* saved as a Git credential."
else
  echo "Error: Hugging Face CLI login failed (Exit code: $LOGIN_STATUS)." >&2
  echo "Please check your token and network connection." >&2
  exit $LOGIN_STATUS # Exit with the same error code as the failed command
fi

# --- Optional: Add other commands that require login ---
# The login state is typically cached temporarily by the CLI
# echo "Running command that requires authentication..."
# huggingface-cli whoami # Example command

exit 0 # Exit successfully