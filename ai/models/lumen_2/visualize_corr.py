import os
import sys
import logging
import matplotlib.pyplot as plt
import boto3

logging.basicConfig(level=logging.INFO)

def get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()

    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing file => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def main():
    remote_s3_key = "data/lumen2/featured/spx_vix_corr_heatmap.png"

    local_filename = "spx_vix_corr_heatmap_downloaded.png"

    download_file_from_s3(remote_s3_key, local_filename)

    logging.info("Displaying correlation heatmap locally...")
    img = plt.imread(local_filename)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Downloaded Correlation Heatmap from S3")
    plt.show()

if __name__ == "__main__":
    main()