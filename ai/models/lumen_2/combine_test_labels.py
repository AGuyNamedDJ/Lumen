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
BUCKET_NAME          = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
SEQUENCES_PREFIX     = "data/lumen2/featured/sequences/"
LABELS_PATTERN       = r"spx_test_Y_3D_part\d+\.npy"  # Regex to match chunk files
LOCAL_CHUNKS_DIR     = "tmp_chunks"                   # local subfolder to store chunk downloads
FINAL_LABELS_LOCAL   = "y_test.npy"
FINAL_LABELS_S3_KEY  = "models/lumen_2/trained/y_test.npy"

##############################################################################
# 3) LIST OBJECTS IN S3
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
# 4) DOWNLOAD + CONCATENATE
##############################################################################
def download_and_combine_y_chunks():
    """
    1. List all spx_test_Y_3D_partX.npy files in S3 (under 'SEQUENCES_PREFIX').
    2. Download them locally.
    3. Load + concatenate into one big array => y_test.npy
    4. Return the combined array shape.
    """
    s3_client = get_s3_client()

    # 1) List all objects
    logging.info(f"Listing S3 objects with prefix => {SEQUENCES_PREFIX}")
    all_objs = list_s3_objects_with_prefix(s3_client, SEQUENCES_PREFIX)

    # 2) Filter for our label pattern (spx_test_Y_3D_partX.npy)
    # Using a regex match:
    chunk_keys = []
    for obj in all_objs:
        key = obj['Key']  # e.g. "data/lumen2/featured/sequences/spx_test_Y_3D_part0.npy"
        filename = os.path.basename(key)
        if re.match(LABELS_PATTERN, filename):
            chunk_keys.append(key)

    if not chunk_keys:
        logging.warning("No spx_test_Y_3D_part*.npy files found => nothing to combine.")
        return None

    logging.info(f"Found {len(chunk_keys)} label chunk file(s):")
    for ck in chunk_keys:
        logging.info(f" - {ck}")

    # 3) Download each chunk
    if not os.path.exists(LOCAL_CHUNKS_DIR):
        os.makedirs(LOCAL_CHUNKS_DIR, exist_ok=True)

    y_parts = []
    for ck in sorted(chunk_keys):
        base = os.path.basename(ck)
        local_path = os.path.join(LOCAL_CHUNKS_DIR, base)

        # Remove local if exists
        if os.path.exists(local_path):
            os.remove(local_path)

        logging.info(f"Downloading s3://{BUCKET_NAME}/{ck} => {local_path}")
        s3_client.download_file(BUCKET_NAME, ck, local_path)
        arr = np.load(local_path)
        y_parts.append(arr)

    # 4) Concatenate
    if not y_parts:
        logging.warning("No arrays to concatenate => empty list.")
        return None

    y_all = np.concatenate(y_parts, axis=0)
    logging.info(f"Final y_all shape => {y_all.shape}")

    # 5) Save local
    np.save(FINAL_LABELS_LOCAL, y_all)
    logging.info(f"Saved => {FINAL_LABELS_LOCAL}")
    return y_all.shape

##############################################################################
# 5) UPLOAD THE COMBINED y_test.npy
##############################################################################
def upload_combined_y_test():
    """
    Uploads 'y_test.npy' => 'models/lumen_2/trained/y_test.npy'
    """
    if not os.path.exists(FINAL_LABELS_LOCAL):
        logging.error(f"File {FINAL_LABELS_LOCAL} not found => cannot upload.")
        return

    s3_client = get_s3_client()
    logging.info(f"Uploading {FINAL_LABELS_LOCAL} => s3://{BUCKET_NAME}/{FINAL_LABELS_S3_KEY}")
    try:
        s3_client.upload_file(FINAL_LABELS_LOCAL, BUCKET_NAME, FINAL_LABELS_S3_KEY)
        logging.info("Upload complete.")
    except Exception as exc:
        logging.error(f"Error uploading y_test.npy => {exc}")

##############################################################################
# 6) MAIN
##############################################################################
def main():
    logging.info("=== combine_test_labels_s3 => Start ===")
    shape = download_and_combine_y_chunks()

    if shape is None:
        logging.error("No label chunks found => aborting.")
        return

    upload_combined_y_test()
    logging.info("=== combine_test_labels_s3 => Done ===")

if __name__ == "__main__":
    main()