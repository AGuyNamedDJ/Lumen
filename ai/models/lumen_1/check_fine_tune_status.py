from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def convert_unix_to_readable(unix_timestamp):
    """
    Convert Unix timestamp to both 24-hour and 12-hour human-readable formats.

    :param unix_timestamp: Unix timestamp.
    :return: Dictionary with 24-hour and 12-hour date strings.
    """
    dt = datetime.utcfromtimestamp(unix_timestamp)
    return {
        '24_hour_format': dt.strftime('%Y-%m-%d %H:%M:%S'),
        '12_hour_format': dt.strftime('%Y-%m-%d %I:%M:%S %p')
    }


def check_fine_tune_status(job_id):
    """
    Check the status of the fine-tuning job.

    :param job_id: ID of the fine-tuning job.
    """
    response = client.fine_tuning.jobs.retrieve(job_id)
    response_dict = response.to_dict()

    # Convert Unix timestamps to readable format
    if response_dict.get("created_at"):
        response_dict["created_at"] = convert_unix_to_readable(
            response_dict["created_at"])
    if response_dict.get("finished_at"):
        response_dict["finished_at"] = convert_unix_to_readable(
            response_dict["finished_at"])
    if response_dict.get("estimated_finish"):
        response_dict["estimated_finish"] = convert_unix_to_readable(
            response_dict["estimated_finish"])

    print("Fine-tuning job status:")
    print(json.dumps(response_dict, indent=4))


def main():
    job_id = 'ftjob-Xh3gQrbWOeSdJvo15QPQ8WJk'  # Update Job IDs
    check_fine_tune_status(job_id)


if __name__ == "__main__":
    main()
