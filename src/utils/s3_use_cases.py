import boto3
import os
import uuid
import json


def check_path(file_path: str):
    """Check if the path is exist in s3 bucket

    Args:
        file_path (str): Full S3 file path, e.g., "s3://my-bucket/path/to/file.txt"
    Returns:
        bool: True if the path exists, False otherwise
    """

    bucket_name, folder_file_path = cut_bucket_name(file_path)

    client = boto3.resource(
        service_name="s3",
        region_name="eu-central-1",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_KEY"),
    )

    objects = client.meta.client.list_objects_v2(Bucket=bucket_name, Prefix="")
    folders = objects["Contents"]
    exist = False
    for f in folders:
        if folder_file_path in f["Key"]:
            exist = True
            break
    return exist


def cut_bucket_name(file_path: str):
    """Cut bucket name from file path and return bucket name and the remaining file path

    Args:
        file_path (str): Full S3 file path, e.g., "s3://my-bucket/path/to/file.txt"
    Returns:
        tuple: (bucket_name, file_path)
    If the path does not contain "s3://", it returns None for bucket_name and the full path.
    """

    if "s3" in file_path:
        file_path = file_path.replace("s3://", "")
        file_path = file_path.split("/")
        bucket_name = file_path[0]
        file_path = "/".join(file_path[1:])
    else:
        bucket_name = None
    return bucket_name, file_path


def upload_file_to_s3(
    file_name, bucket, path, file_format="txt", region_name="eu-central-1"
):
    """
    Upload a file to an S3 bucket

    file_name: File to upload
    bucket: Bucket to upload to
    path: Path to upload to in the bucket
    file_format: Format of the file
    region_name: AWS region name

    Returns: URL of the uploaded file
    """

    with open(file_name, "r") as file:
        data = json.load(file)
        if data is None:
            return "No file found!"
        elif len(data) == []:
            return "File is empty!"

    new_file_name = (
        file_name.rsplit(".")[0] + "_" + str(uuid.uuid4()) + "." + file_format
    )

    if path[-1] != "/":
        path = path + "/"

    object_name = path + new_file_name
    s3 = boto3.resource(
        service_name="s3",
        region_name=region_name,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
    )
    s3.meta.client.upload_file(Filename=file_name, Bucket=bucket, Key=object_name)

    return "https://{bucket}.s3.{region_name}.amazonaws.com/{path}/{file_name}".format(
        bucket=bucket, region_name=region_name, path=path, file_name=new_file_name
    )
