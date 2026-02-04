#!/bin/bash
# Launch multiple AWS instances for parallel evaluation
# Instance 1: Basic quality + complexity evaluation
# Instance 2-3: Nguyen suite with RL (split 1-6 and 7-12)

set -e

HF_TOKEN="${1:-}"
WANDB_KEY="${2:-}"

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
    echo "Usage: $0 <HF_TOKEN> <WANDB_KEY>"
    exit 1
fi

INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-0c2b0d3d5d8a8a0a0"  # Deep Learning AMI
KEY_NAME="chave-gpu"
SECURITY_GROUP="sg-0deaa73e23482e3f6"

# Auto-detect AMI
echo "Auto-detecting Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  "Name=state,Values=available" \
  --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
  --output text \
  --region us-east-1)

echo "Using AMI: $AMI_ID"

# Function to create userdata script
create_userdata() {
    local instance_name=$1
    cat > aws/temp/userdata_${instance_name}.sh << 'USERDATA_EOF'
#!/bin/bash
set -x
exec > >(tee -a /home/ubuntu/setup.log) 2>&1

echo "===== Starting Setup ====="
date

# Update system
apt-get update
apt-get install -y git python3-pip python3-venv htop

# Clone repo
cd /home/ubuntu
sudo -u ubuntu git clone https://github.com/augustocsc/seriguela.git || true
cd seriguela
sudo -u ubuntu git pull

# Setup venv
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu bash -c "source venv/bin/activate && pip install --upgrade pip"
sudo -u ubuntu bash -c "source venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu121"
sudo -u ubuntu bash -c "source venv/bin/activate && pip install -r requirements.txt"

# Set credentials
echo "export HF_TOKEN=PLACEHOLDER_HF_TOKEN" >> /home/ubuntu/.bashrc
echo "export WANDB_API_KEY=PLACEHOLDER_WANDB_KEY" >> /home/ubuntu/.bashrc

# Create directories
sudo -u ubuntu mkdir -p /home/ubuntu/seriguela/output
sudo -u ubuntu mkdir -p /home/ubuntu/seriguela/results

echo "===== Setup Complete ====="
touch /home/ubuntu/.setup_complete
date
USERDATA_EOF

    # Replace placeholders
    sed -i "s/PLACEHOLDER_HF_TOKEN/$HF_TOKEN/g" aws/temp/userdata_${instance_name}.sh
    sed -i "s/PLACEHOLDER_WANDB_KEY/$WANDB_KEY/g" aws/temp/userdata_${instance_name}.sh
}

# Launch instances
launch_instance() {
    local name=$1
    echo ""
    echo "Launching instance: $name"

    create_userdata "$name"

    INSTANCE_ID=$(aws ec2 run-instances \
      --image-id $AMI_ID \
      --instance-type $INSTANCE_TYPE \
      --key-name $KEY_NAME \
      --security-group-ids $SECURITY_GROUP \
      --user-data file://aws/temp/userdata_${name}.sh \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=seriguela-${name}}]" \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
      --query 'Instances[0].InstanceId' \
      --output text)

    echo "Instance launched: $INSTANCE_ID"
    echo "$name=$INSTANCE_ID" >> aws/temp/instance_ids.txt
}

# Clear old instance IDs
rm -f aws/temp/instance_ids.txt

# Launch all instances in parallel
echo "=========================================="
echo "Launching 3 AWS instances in parallel"
echo "=========================================="

launch_instance "eval-basic" &
launch_instance "nguyen-1-6" &
launch_instance "nguyen-7-12" &

wait

echo ""
echo "All instances launched!"
echo ""
cat aws/temp/instance_ids.txt
echo ""
echo "Waiting for instances to be running..."

# Get all instance IDs
INSTANCE_IDS=$(cat aws/temp/instance_ids.txt | cut -d'=' -f2 | tr '\n' ' ')

aws ec2 wait instance-running --instance-ids $INSTANCE_IDS

echo ""
echo "All instances are running!"
echo ""
echo "Getting public IPs..."

for line in $(cat aws/temp/instance_ids.txt); do
    name=$(echo $line | cut -d'=' -f1)
    id=$(echo $line | cut -d'=' -f2)
    ip=$(aws ec2 describe-instances \
      --instance-ids $id \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text)
    echo "$name: $ip (ID: $id)"
    echo "$name=$ip" >> aws/temp/instance_ips.txt
done

echo ""
echo "=========================================="
echo "Instances ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Wait ~3 minutes for setup to complete"
echo "2. Upload models to instances"
echo "3. Start evaluations"
echo ""
