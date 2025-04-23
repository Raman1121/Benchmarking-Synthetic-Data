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
# Ensure the token file *only* contains the token, ideally without a trailing newline,
# although 'huggingface-cli' is usually tolerant of that.
# Using double quotes around "$(cat ...)" is good practice, though less critical here.
HF_TOKEN=$(cat "$TOKEN_FILE")

# 3. Check if the token was read successfully (is not empty)
if [ -z "$HF_TOKEN" ]; then
  echo "Error: Token file '$TOKEN_FILE' appears to be empty." >&2
  exit 1
fi

# 4. Log in using the token
echo "Attempting to log in to Hugging Face CLI..."
# Use the --token argument to provide the token non-interactively
# The --add-to-git-credential flag is often useful to configure git credential helper
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# 5. Check the exit status of the login command
LOGIN_STATUS=$? # Capture the exit code of the last command
if [ $LOGIN_STATUS -eq 0 ]; then
  echo "Hugging Face CLI login successful."
else
  echo "Error: Hugging Face CLI login failed (Exit code: $LOGIN_STATUS)." >&2
  echo "Please check your token and network connection." >&2
  exit $LOGIN_STATUS # Exit with the same error code as the failed command
fi

# --- Optional: Add other commands that require login ---
# echo "Running command that requires authentication..."
# hf-transfer download ...

exit 0 # Exit successfully