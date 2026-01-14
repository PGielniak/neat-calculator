from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os

def download_blob_to_dir(storage_account_blob_uri: str, download_dir: str):
    """
    Downloads a blob from Azure Storage to a local directory.

    Args:
        storage_account_blob_uri (str): The full URI of the blob in Azure Storage.
        download_dir (str): The local directory to download the blob to.
    """
    
    ACCOUNT_NAME = storage_account_blob_uri.split('@')[1].split('.')[0]
    CONTAINER = storage_account_blob_uri.split('://')[1].split('@')[0]
    BLOB_PATH = '/'.join(storage_account_blob_uri.split('://')[1].split('@')[1].split('/')[1:])

    account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=BLOB_PATH)

    os.makedirs(download_dir, exist_ok=True)
    download_file_path = os.path.join(download_dir, os.path.basename(BLOB_PATH))

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    print(f"Downloaded blob to {download_file_path}")