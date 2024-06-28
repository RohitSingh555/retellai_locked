import boto3
import re
import os

s3 = boto3.client('s3', region_name='us-east-1') 

bucket_name = 'sample-candidates-pluto-dev'

prefix = 'user_data/'

local_folder = '../transcripts/'

def file_exists_in_s3(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        return False

def move_files_to_user_folders(local_folder, bucket_name, prefix):
    for file_name in os.listdir(local_folder):
        local_path = os.path.join(local_folder, file_name)
        
        if os.path.isfile(local_path):
            match = re.match(r'(\d+)_', file_name)
            if match:
                user_id = match.group(1)
                new_key = f"{prefix}{user_id}/{file_name}"
                
                if not file_exists_in_s3(bucket_name, new_key):
                    s3.upload_file(local_path, bucket_name, new_key)
                    
                    print(f"Uploaded {local_path} to s3://{bucket_name}/{new_key}")
                else:
                    print(f"File {file_name} already exists in s3://{bucket_name}/{prefix}{user_id}/")
            else:
                print(f"No user id found in {file_name}")

move_files_to_user_folders(local_folder, bucket_name, prefix)
