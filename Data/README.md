You have two options:

1. Download the data directly from the S3 bucket using the provided scripts.
2. Manually upload your own (clean) data and use the provided scripts to process it.

In details:

# Option 1: Download from S3

1. Make sure you have the AWS CLI installed and configured.
2. Run the provided download scripts to fetch the data from the S3 bucket.

Namely you can copy paste these commands:

```bash
# Download Image Data
bash download_images_data_s3.sh

# Download Prolific Data
bash download_prolific_data_s3.sh
```

Will create two folders:

- `Data/Image_Data/` : containing all the clean and corrupted images + DeepDream Generated
- `Data/Prolific_Data/` : containing all the Prolific data files

# Option 2: Upload and Process Your Own Data

1. Prepare your clean data files and place them in 'Data/Image_Data/clean_images'.
2. Follow instructions in DIY.
