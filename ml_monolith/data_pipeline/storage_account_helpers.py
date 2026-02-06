from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os
import logging

#TODO: Add error handling and logging
#TODO: test with different storage account configurations
def download_blob_to_dir(storage_account_blob_uri: str, download_dir: str, logger: logging.Logger = None) -> None:
    """
    Downloads blob(s) from Azure Storage to a local directory.
    If the URI points to a prefix/folder, downloads all blobs with that prefix.
    If it points to a single blob, downloads just that blob.

    Args:
        storage_account_blob_uri (str): The full URI of the blob or prefix in Azure Storage.
        download_dir (str): The local directory to download the blob(s) to.
        logger (logging.Logger): Optional logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    ACCOUNT_NAME = storage_account_blob_uri.split('@')[1].split('.')[0]
    CONTAINER = storage_account_blob_uri.split('://')[1].split('@')[0]
    
    # Extract the path after the domain, then remove the container name if it's duplicated
    path_after_domain = '/'.join(storage_account_blob_uri.split('://')[1].split('@')[1].split('/')[1:])
    
    # If the path starts with the container name again, remove it
    if path_after_domain.startswith(CONTAINER + '/'):
        BLOB_PATH = path_after_domain[len(CONTAINER) + 1:]
    else:
        BLOB_PATH = path_after_domain
    
    # Remove trailing slash if present
    BLOB_PATH = BLOB_PATH.rstrip('/')
    
    logger.debug(f"Storage Account: {ACCOUNT_NAME}, Container: {CONTAINER}, Blob Path: {BLOB_PATH}")
    
    account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
    
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        logger.info("Using AZURE_STORAGE_CONNECTION_STRING for authentication.")
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
    else:
        try:
            logger.info("Using DefaultAzureCredential for authentication.")
            credential = DefaultAzureCredential()
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        except Exception as e:
            logger.error(f"Failed to authenticate with DefaultAzureCredential: {e}")
            raise e
    
    container_client = blob_service_client.get_container_client(CONTAINER)
    
    logger.info(f"Listing blobs with prefix: {BLOB_PATH}")
    
    os.makedirs(download_dir, exist_ok=True)
    
    # First, try to download as a single blob
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=BLOB_PATH)
        # It's a single blob file
        download_file_path = os.path.join(download_dir, os.path.basename(BLOB_PATH))
        logger.info(f"Downloading single blob {BLOB_PATH} to {download_file_path}")
        
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        logger.info(f"Downloaded 1 blob from {storage_account_blob_uri}")
        return
    except Exception as e:
        logger.debug(f"Not a single blob, treating as prefix: {e}")
    
    # If not a single blob, list all blobs with the given prefix
    blob_list = container_client.list_blobs(name_starts_with=BLOB_PATH)
    
    downloaded_count = 0
    for blob in blob_list:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=blob.name)
        
        # Preserve directory structure within download_dir
        relative_path = blob.name[len(BLOB_PATH):].lstrip('/')
        if not relative_path:
            relative_path = os.path.basename(blob.name)
        
        download_file_path = os.path.join(download_dir, relative_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        
        logger.info(f"Downloading {blob.name} to {download_file_path}")
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        downloaded_count += 1
    
    logger.info(f"Downloaded {downloaded_count} blob(s) from {storage_account_blob_uri}")