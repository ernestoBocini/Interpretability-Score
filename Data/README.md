# Data Setup

You have two options:

1. Download the data directly from the S3 bucket using the provided scripts.
2. Manually upload your own (clean) data and use the provided scripts to process it.

## Option 1: Download from S3

### Prerequisites
- AWS CLI installed and configured
- Access to the `ernesto-interpretability` S3 bucket

### Download Commands

Run the provided download scripts to fetch all datasets:

```bash
# Download Image Data (39.5 MiB, 1262 files)
./download_images_data_s3.sh

# Download Prolific Data
./download_prolific_s3.sh

# Download CLIP Activation Layer Data
./download_clip_activation_s3.sh
```

### Output Structure

This will create three folders:

- `Images_Data/` : containing all the clean and corrupted images + DeepDream Generated
- `Prolific_Data/` : containing all the Prolific experiment data files
- `Clip_Activation_Layer/` : containing CLIP activation layer data

### AWS Setup

If you haven't configured AWS CLI yet:

1. **For IAM Identity Center (SSO) users:**
   ```bash
   aws configure sso
   aws sso login
   ```

2. **For users with access keys:**
   ```bash
   aws configure
   ```

3. **For temporary credentials:**
   - Go to your AWS Access Portal
   - Select account → "Command line or programmatic access"
   - Copy and run the export commands

### Troubleshooting

- **"Unknown options: --progress"**: The scripts automatically handle different AWS CLI versions
- **Access denied**: Ensure your AWS credentials are valid and you have S3 bucket permissions
- **Credentials expired**: For SSO users, run `aws sso login` again

## Option 2: Upload and Process Your Own Data

1. Prepare your clean data files and place them in `Images_Data/clean_images/`
2. Follow instructions in DIY section

---

**Note**: All download scripts include error checking, progress reporting, and will resume interrupted downloads automatically.