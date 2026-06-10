# AWS Configuration for Seriguela

This directory contains AWS-related configuration and credentials for running training instances.

## Directory Structure

```
aws/
├── README.md           # This file
├── config.json         # Instance configuration (safe to commit)
├── keys/              # SSH keys directory (gitignored)
└── .env               # Environment variables (gitignored)

**Note**: SSH key is located at `~/chave-gpu.pem` (not in this directory)
```

## Setup Instructions

### 1. AWS CLI Configuration

Make sure AWS CLI is configured with your credentials:

```bash
aws configure
```

Or manually create `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

### 2. SSH Key Setup

1. Download your EC2 key pair from AWS Console or create a new one:
   ```bash
   aws ec2 create-key-pair --key-name chave-gpu --query 'KeyMaterial' --output text > ~/chave-gpu.pem
   ```

2. Set proper permissions:
   ```bash
   chmod 400 ~/chave-gpu.pem
   ```

### 3. Security Group Configuration

Current security group: `sg-0deaa73e23482e3f6` (seriguela-sg)

To add your IP for SSH access:
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0deaa73e23482e3f6 \
  --protocol tcp \
  --port 22 \
  --cidr $MY_IP/32
```

### 4. Environment Variables (Optional)

Create `aws/.env` for project-specific settings:

```bash
AWS_REGION=us-east-1
AWS_KEY_NAME=chave-gpu
AWS_SECURITY_GROUP=sg-0deaa73e23482e3f6
```

## Usage

### Launch an Instance

```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g5.xlarge \
  --key-name chave-gpu \
  --security-group-ids sg-0deaa73e23482e3f6 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=seriguela-training}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}'
```

### Connect to Instance

```bash
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-training" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

# SSH connect
ssh -i ~/chave-gpu.pem ubuntu@$INSTANCE_IP
```

### Stop All Running Instances

```bash
aws ec2 stop-instances \
  --instance-ids $(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=seriguela-training" "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[*].InstanceId" \
    --output text)
```

### List All Instances

```bash
aws ec2 describe-instances \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" \
  --output table
```

## Configuration Profiles

The `config.json` file contains different profiles:

- **default**: Standard g5.xlarge instance for regular training
- **training**: Spot instance configuration for cost savings
- **large_training**: p4d.24xlarge for large-scale training

## Security Best Practices

1. **Never commit** `.pem` files or AWS credentials to git
2. Use **IAM roles** for EC2 instances instead of hardcoded credentials
3. Regularly **rotate** your access keys
4. Use **Security Groups** to restrict access by IP
5. Always **stop instances** when not in use to avoid charges
6. Consider using **AWS Systems Manager Session Manager** instead of SSH

## Cost Management

- **g5.xlarge**: ~$1.006/hour (on-demand)
- **p4d.24xlarge**: ~$32.77/hour (on-demand)
- Use **spot instances** for training to save up to 70%
- Set up **billing alerts** in AWS Console

## Current Resources

- **Account ID**: 452379801849
- **Region**: us-east-1
- **Security Group**: sg-0deaa73e23482e3f6 (seriguela-sg)
- **Allowed IPs**: 143.106.58.120/32, 179.160.37.193/32
