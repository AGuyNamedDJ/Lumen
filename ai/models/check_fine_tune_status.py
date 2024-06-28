from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def check_fine_tune_status(job_id):
    """
    Check the status of the fine-tuning job.

    :param job_id: ID of the fine-tuning job.
    """
    response = client.fine_tuning.jobs.retrieve(job_id)
    print("Fine-tuning job status:")
    # Format error, if not present its unreadable
    print(json.dumps(response.to_dict(), indent=4))


def main():
    job_id = 'ftjob-Xh3gQrbWOeSdJvo15QPQ8WJk'  # Update Job IDs
    check_fine_tune_status(job_id)


if __name__ == "__main__":
    main()
