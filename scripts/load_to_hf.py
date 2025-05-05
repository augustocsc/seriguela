# upload_dataset_to_hf.py

import argparse
import os
import sys
import subprocess
from datasets import load_dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, HfFolder, login, HfApi
# Added import for HfFolder

# --- Helper Function to Check Git LFS ---
def check_git_lfs_installed():
    """Checks if git-lfs is installed and configured."""
    try:
        # Check if git-lfs command exists
        subprocess.run(["git", "lfs", "--version"], check=True, capture_output=True)
        # Check if git-lfs is initialized for the user (optional but good practice)
        # This command might vary or not be strictly necessary depending on setup
        # subprocess.run(["git", "config", "--global", "--get", "filter.lfs.smudge"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: git-lfs command not found or not configured.")
        print("         Please install git-lfs and run 'git lfs install --system' (or --user).")
        print("         See: https://git-lfs.com/")
        # Optionally exit if git-lfs is strictly required
        # sys.exit(1)
        return False # Allow script to continue but warn user

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Upload CSV dataset splits from a local directory to the Hugging Face Hub."
    )

    # --- Required Arguments ---
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Path to the local directory containing the dataset CSV files."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The Hugging Face Hub repository ID (e.g., 'username/my-equation-dataset')."
    )
    parser.add_argument(
        "--data_column",
        type=str,
        required=True,
        help="Name of the column in the CSV files containing the actual data (e.g., 'text', 'equation')."
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "--train_filename",
        type=str,
        default=None,
        help="Filename of the training CSV within local_dir (e.g., 'train_data.csv')."
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default=None,
        help="Filename of the validation CSV within local_dir (e.g., 'validation_set.csv')."
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default=None,
        help="Filename of the test CSV within local_dir (optional, e.g., 'test_examples.csv')."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Your Hugging Face Hub access token (with write permissions). If not provided, script will try to use cached token or prompt login."
    )
    parser.add_argument(
        "--private",
        action='store_true', # Makes the repo private if flag is present
        help="Set the Hugging Face repository to private."
    )

    args = parser.parse_args()

    print("--- Starting Dataset Upload Script ---")

    # 1. Check Git LFS
    print("Checking for git-lfs...")
    check_git_lfs_installed() # Warns if not found

    # 2. Handle Authentication
    token = args.hf_token
    if not token:
        token = HfFolder.get_token() # Try to get cached token

    if not token:
        print("\nAttempting Hugging Face login...")
        try:
            login() # Will prompt user if not logged in via CLI
            token = HfFolder.get_token() # Get token after successful login
            if not token:
                 raise Exception("Login seemed successful but token could not be retrieved.")
        except Exception as e:
            print(f"Error during Hugging Face login: {e}")
            print("Please ensure you are logged in via 'huggingface-cli login' or provide a token using --hf_token.")
            sys.exit(1)
    else:
         print("Using provided/cached Hugging Face token.")
         # Optionally verify token validity here if needed, though push_to_hub will fail if invalid


    # 3. Determine Filenames
    dir_name = os.path.basename(os.path.normpath(args.local_dir)) # Gets the last part of the path

    train_file = args.train_filename if args.train_filename else f"train_{dir_name}.csv"
    val_file = args.val_filename if args.val_filename else f"val_{dir_name}.csv" # Using 'val' as abbreviation
    test_file = args.test_filename if args.test_filename else f"test_{dir_name}.csv"

    print(f"Using directory: {args.local_dir}")
    print(f"Target Hub repo: {args.repo_id}")
    print(f"Expecting data column: '{args.data_column}'")
    print(f"Using train file: '{train_file}'")
    print(f"Using validation file: '{val_file}'")
    # Test file is optional, only check if default or specific name provided
    if args.test_filename or os.path.exists(os.path.join(args.local_dir, test_file)):
         print(f"Using test file: '{test_file}'")
    else:
        print("No test file specified or default test file not found, skipping.")
        test_file = None # Ensure test_file is None if not used


    # 4. Construct Full Paths and Check Existence
    train_path = os.path.join(args.local_dir, train_file)
    val_path = os.path.join(args.local_dir, val_file)
    test_path = os.path.join(args.local_dir, test_file) if test_file else None

    data_files = {}
    if os.path.exists(train_path):
        data_files["train"] = train_path
    else:
        print(f"Error: Training file not found at '{train_path}'")
        sys.exit(1)

    if os.path.exists(val_path):
        data_files["validation"] = val_path
    else:
        print(f"Error: Validation file not found at '{val_path}'")
        sys.exit(1)

    if test_path and os.path.exists(test_path):
        data_files["test"] = test_path
    elif args.test_filename: # If user specified a test file but it wasn't found
         print(f"Warning: Specified test file '{args.test_filename}' not found at '{test_path}'. Skipping test split.")


    # 5. Load Dataset Locally
    print("\nLoading local CSV files...")
    try:
        # Define features to ensure the data column is read as string
        features = Features({args.data_column: Value('string')})
        dataset_dict = load_dataset("csv", data_files=data_files, features=features)
        print("Local dataset loaded successfully:")
        print(dataset_dict)

        # Verify the data column exists in the loaded dataset
        for split in dataset_dict:
             if args.data_column not in dataset_dict[split].column_names:
                  print(f"Error: Column '{args.data_column}' not found in loaded '{split}' split.")
                  print(f"Available columns: {dataset_dict[split].column_names}")
                  sys.exit(1)

    except Exception as e:
        print(f"Error loading dataset from CSV files: {e}")
        print("Please check file paths, CSV format, and column names.")
        sys.exit(1)

    # 6. Rename column if necessary (optional, often good to standardize to 'text')
    # If you always want the main data column to be named 'text' on the Hub:
    if args.data_column != 'text':
         print(f"Renaming column '{args.data_column}' to 'text'...")
         try:
             dataset_dict = dataset_dict.rename_column(args.data_column, "text")
             print("Column renamed successfully.")
             print(dataset_dict)
         except Exception as e:
             print(f"Error renaming column: {e}")
             # Decide if you want to exit or proceed with the original column name
             # sys.exit(1)


    # 7. Push to Hub
    print(f"\nAttempting to push dataset to Hub repository: {args.repo_id}...")
    try:
        dataset_dict.push_to_hub(
            repo_id=args.repo_id,
            private=args.private,
            token=token # Pass token explicitly
            )
        print("\n--- Upload Successful! ---")
        hub_url = f"https://huggingface.co/datasets/{args.repo_id}"
        print(f"Dataset available at: {hub_url}")

    except Exception as e:
        print(f"\n--- Error During Upload ---")
        print(f"An error occurred: {e}")
        print("Possible causes:")
        print("- Invalid Hugging Face token or insufficient permissions (needs write access).")
        print("- Repository ID format incorrect (should be 'username/dataset_name').")
        print("- Network issues.")
        print("- Git LFS not installed or properly configured.")
        print("- Conflicts if the repository already exists with incompatible content.")
        sys.exit(1)

if __name__ == "__main__":
    main()