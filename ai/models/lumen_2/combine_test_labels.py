import os
import re
import sys
import logging
import boto3
import numpy as np
from dotenv import load_dotenv

##############################################################################
# 1) AWS CONFIG + LOGGING
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

def get_s3_client():
    """
    Creates a boto3 S3 client pinned to region us-east-2 (or whichever region you use),
    pulling credentials from environment variables.
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

##############################################################################
# 2) CONSTANTS
##############################################################################
BUCKET_NAME           = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
SEQUENCES_PREFIX      = "data/lumen2/featured/sequences/"
LOCAL_CHUNKS_DIR      = "tmp_chunks"

# -- Y (Labels) --
Y_PATTERN             = r"spx_test_Y_3D_part\d+\.npy" 
FINAL_LABELS_LOCAL    = "y_test.npy"
FINAL_LABELS_S3_KEY   = "models/lumen_2/trained/y_test.npy"

# -- X (Features) --
X_PATTERN             = r"spx_test_X_3D_part\d+\.npy"
FINAL_FEATURES_LOCAL  = "X_test.npy"
FINAL_FEATURES_S3_KEY = "models/lumen_2/trained/X_test.npy"

##############################################################################
# 3) HELPER: LIST OBJECTS IN S3
##############################################################################
def list_s3_objects_with_prefix(s3_client, prefix):
    """
    Lists all objects in 'BUCKET_NAME' that start with 'prefix'.
    Returns a list of dict (S3 item info).
    """
    results = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                results.append(obj)
    return results

##############################################################################
# 4) GENERIC: Download + Combine Chunks
##############################################################################
def download_and_combine_chunks(pattern, final_local_filename):
    """
    1. Lists all .npy files under SEQUENCES_PREFIX that match `pattern` (regex).
    2. Downloads each chunk locally into LOCAL_CHUNKS_DIR.
    3. Loads + concatenates them into one big array => final_local_filename (e.g. "X_test.npy" or "y_test.npy").
    4. Returns the final array shape or None if none found.
    """
    s3_client = get_s3_client()
    logging.info(f"Listing S3 objects with prefix => {SEQUENCES_PREFIX}")
    all_objs = list_s3_objects_with_prefix(s3_client, SEQUENCES_PREFIX)

    chunk_keys = []
    for obj in all_objs:
        key = obj["Key"]  
        filename = os.path.basename(key)
        if re.match(pattern, filename):
            chunk_keys.append(key)

    if not chunk_keys:
        logging.warning(f"No files found matching pattern {pattern} => nothing to combine.")
        return None

    logging.info(f"Found {len(chunk_keys)} chunk file(s) matching {pattern}:")
    for ck in chunk_keys:
        logging.info(f" - {ck}")

    if not os.path.exists(LOCAL_CHUNKS_DIR):
        os.makedirs(LOCAL_CHUNKS_DIR, exist_ok=True)

    parts = []
    for ck in sorted(chunk_keys):
        base = os.path.basename(ck)
        local_path = os.path.join(LOCAL_CHUNKS_DIR, base)

        if os.path.exists(local_path):
            os.remove(local_path)

        logging.info(f"Downloading s3://{BUCKET_NAME}/{ck} => {local_path}")
        s3_client.download_file(BUCKET_NAME, ck, local_path)
        arr = np.load(local_path)
        parts.append(arr)

    if not parts:
        logging.warning("No arrays to concatenate => empty list.")
        return None

    combined = np.concatenate(parts, axis=0)
    logging.info(f"Final combined shape => {combined.shape}")

    np.save(final_local_filename, combined)
    logging.info(f"Saved => {final_local_filename}")
    return combined.shape

##############################################################################
# 5) UPLOAD COMBINED FILE
##############################################################################
def upload_file_to_s3(local_path, s3_key):
    s3_client = get_s3_client()
    if not os.path.exists(local_path):
        logging.error(f"File {local_path} not found => cannot upload.")
        return
    logging.info(f"Uploading {local_path} => s3://{BUCKET_NAME}/{s3_key}")
    try:
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        logging.info("Upload complete.")
    except Exception as exc:
        logging.error(f"Error uploading {local_path} => {exc}")

##############################################################################
# 6) MAIN
##############################################################################
def main():
    logging.info("=== combine_test_labels_s3 => Start ===")

    # -- Combine Y (labels) --
    shape_y = download_and_combine_chunks(Y_PATTERN, FINAL_LABELS_LOCAL)
    if shape_y is not None:
        upload_file_to_s3(FINAL_LABELS_LOCAL, FINAL_LABELS_S3_KEY)

    # -- Combine X (features) --
    shape_x = download_and_combine_chunks(X_PATTERN, FINAL_FEATURES_LOCAL)
    if shape_x is not None:
        upload_file_to_s3(FINAL_FEATURES_LOCAL, FINAL_FEATURES_S3_KEY)

    logging.info("=== combine_test_labels_s3 => Done ===")

if __name__ == "__main__":
    main()