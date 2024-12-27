import os
import sys
import logging
import boto3
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')

def get_s3_client():
    """
    Returns a boto3 S3 client object using the AWS creds in .env,
    or fallback to environment variables set on the OS.
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def upload_file_to_s3(local_path: str, s3_key: str):
    """
    Basic utility to upload a file to S3 if you already know s3_key.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"Uploaded {local_path} → s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Error uploading {local_path} to S3: {e}")

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Downloads a file from S3 to your local filesystem.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from S3: {e}")


def auto_upload_file_to_s3(local_path: str, s3_subfolder: str = ""):
    """
    Uploads a file to S3, automatically deriving its filename
    from `local_path` and placing it in an optional `s3_subfolder`.

    :param local_path: The local path, e.g. "/path/to/file.csv"
    :param s3_subfolder: If provided, the subfolder in your S3 bucket
                         (e.g. "data/lumen2/featured" or "models/lumen_2/scalers").
                         If empty, uploads to the bucket root.
    """
    import os

    # 1) Derive local filename:
    local_filename = os.path.basename(local_path) 

    # 2) Build the final S3 key
    if s3_subfolder:
        s3_key = f"{s3_subfolder}/{local_filename}"
    else:
        s3_key = local_filename

    # 3) Actually call the simpler upload method
    upload_file_to_s3(local_path, s3_key)