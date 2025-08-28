#!/bin/bash

# S3 Download Script for Prolific Data
# Bucket: ernesto-interpretability
# Folder: prolific_data/

set -e  # Exit on any error

# Configuration
BUCKET_NAME="ernesto-interpretability"
FOLDER_PATH="prolific_data/"
LOCAL_DOWNLOAD_DIR="./Prolific_Data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}S3 Prolific Data Download Script${NC}"
echo "Bucket: s3://${BUCKET_NAME}"
echo "Folder: ${FOLDER_PATH}"
echo "Local destination: ${LOCAL_DOWNLOAD_DIR}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    echo "Please install AWS CLI first:"
    echo "  - macOS: brew install awscli"
    echo "  - Ubuntu/Debian: sudo apt install awscli"
    echo "  - Or visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured or invalid${NC}"
    echo "Please configure your AWS credentials first:"
    echo "  aws configure"
    echo ""
    echo "Or for SSO users:"
    echo "  aws sso login"
    echo ""
    echo "Or set environment variables:"
    echo "  export AWS_ACCESS_KEY_ID=your_access_key"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "  export AWS_SESSION_TOKEN=your_session_token  # if using temporary credentials"
    echo "  export AWS_DEFAULT_REGION=your_region"
    exit 1
fi

# Test bucket access
echo -e "${YELLOW}Testing bucket access...${NC}"
if ! aws s3 ls "s3://${BUCKET_NAME}/" &> /dev/null; then
    echo -e "${RED}Error: Cannot access bucket s3://${BUCKET_NAME}/${NC}"
    echo "This could be due to:"
    echo "  1. Insufficient permissions"
    echo "  2. Incorrect bucket name"
    echo "  3. Bucket doesn't exist"
    echo "  4. Expired SSO/temporary credentials"
    echo ""
    echo "If using SSO, try: aws sso login"
    echo "Please contact the bucket owner to request access."
    exit 1
fi

# Check if folder exists
echo -e "${YELLOW}Checking if folder exists...${NC}"
if ! aws s3 ls "s3://${BUCKET_NAME}/${FOLDER_PATH}" &> /dev/null; then
    echo -e "${RED}Error: Folder ${FOLDER_PATH} not found in bucket${NC}"
    echo ""
    echo "Available folders in bucket:"
    aws s3 ls "s3://${BUCKET_NAME}/" || echo "Cannot list bucket contents"
    exit 1
fi

# Create local directory
echo -e "${YELLOW}Creating local directory...${NC}"
mkdir -p "${LOCAL_DOWNLOAD_DIR}"

# Get folder size for progress indication
echo -e "${YELLOW}Calculating download size...${NC}"
TOTAL_SIZE=$(aws s3 ls "s3://${BUCKET_NAME}/${FOLDER_PATH}" --recursive --human-readable --summarize | grep "Total Size" | awk '{print $3 " " $4}' || echo "unknown")
TOTAL_OBJECTS=$(aws s3 ls "s3://${BUCKET_NAME}/${FOLDER_PATH}" --recursive --summarize | grep "Total Objects" | awk '{print $3}' || echo "unknown")

echo "Total size to download: ${TOTAL_SIZE}"
echo "Total objects: ${TOTAL_OBJECTS}"
echo ""

# Download the folder
echo -e "${GREEN}Starting download...${NC}"
echo "Command: aws s3 sync s3://${BUCKET_NAME}/${FOLDER_PATH} ${LOCAL_DOWNLOAD_DIR}/"
echo ""

# Use sync for better performance and resume capability
# Note: --progress flag is only available in newer AWS CLI versions
AWS_CLI_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
MAJOR_VERSION=$(echo $AWS_CLI_VERSION | cut -d. -f1)

if [ "$MAJOR_VERSION" -ge 2 ] && aws s3 sync --help | grep -q "\--progress" 2>/dev/null; then
    # AWS CLI v2 with progress support
    aws s3 sync "s3://${BUCKET_NAME}/${FOLDER_PATH}" "${LOCAL_DOWNLOAD_DIR}/" \
        --progress \
        --no-follow-symlinks
else
    # AWS CLI v1 or v2 without progress support
    echo "Note: Using basic sync (no progress bar available in this AWS CLI version)"
    aws s3 sync "s3://${BUCKET_NAME}/${FOLDER_PATH}" "${LOCAL_DOWNLOAD_DIR}/" \
        --no-follow-symlinks
fi

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Prolific data download completed successfully!${NC}"
    echo "Files downloaded to: ${LOCAL_DOWNLOAD_DIR}/"
    echo ""
    echo "Downloaded files:"
    find "${LOCAL_DOWNLOAD_DIR}" -type f | head -10
    if [ $(find "${LOCAL_DOWNLOAD_DIR}" -type f | wc -l) -gt 10 ]; then
        echo "... and $(( $(find "${LOCAL_DOWNLOAD_DIR}" -type f | wc -l) - 10 )) more files"
    fi
else
    echo -e "${RED}Download failed!${NC}"
    exit 1
fi