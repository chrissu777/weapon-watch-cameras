import boto3
from datetime import datetime

import aws_encryption_sdk
from aws_encryption_sdk.identifiers import CommitmentPolicy
from aws_encryption_sdk.streaming_client import StreamEncryptor

from aws_cryptographic_material_providers.mpl.models import CreateAwsKmsKeyringInput
from aws_cryptographic_material_providers.mpl import AwsCryptographicMaterialProviders
from aws_cryptographic_material_providers.mpl.config import MaterialProvidersConfig
from aws_cryptographic_material_providers.mpl.models import CreateAwsKmsKeyringInput
from aws_cryptographic_material_providers.mpl.references import IKeyring

from cloud.config import KMS_KEY_ARN

S3_BUCKET = 'weaponwatch-demo'
s3_client = boto3.client('s3')

def encrypt_and_upload(file_path, s3_key, cam_name):
    """Encrypts file using AWS KMS and uploads to provided S3 bucket"""
    
    # Initialize AWS Encryption SDK client
    client = aws_encryption_sdk.EncryptionSDKClient(
        commitment_policy=CommitmentPolicy.REQUIRE_ENCRYPT_REQUIRE_DECRYPT
    )

    # Create KMS client for encryption
    kms_client = boto3.client('kms', region_name="us-east-1")

    # Configure cryptographic material providers
    mat_prov: AwsCryptographicMaterialProviders = AwsCryptographicMaterialProviders(
        config=MaterialProvidersConfig()
    )

    # Create AWS KMS keyring for encryption
    keyring_input: CreateAwsKmsKeyringInput = CreateAwsKmsKeyringInput(
        kms_key_id=KMS_KEY_ARN,
        kms_client=kms_client
    )
    kms_keyring: IKeyring = mat_prov.create_aws_kms_keyring(
        input=keyring_input
    )

    try:
        # Read file and encrypt contents
        with open(file_path, "rb") as infile:
            ciphertext, _ = client.encrypt(
                source=infile,
                keyring=kms_keyring,
            )

        # Upload encrypted data to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=bytes(ciphertext))
        
        print(f"\nENCRYPTED FILE UPLOADED SUCCESSFULLY: {s3_key}")
        
        formatted_time = datetime.now().strftime("%H:%M:%S")
        print(f"UPLOADED AT {formatted_time} FOR {cam_name}")

    except Exception as e:
        print(f"Error: {str(e)}")  # Print error if encryption or upload fails