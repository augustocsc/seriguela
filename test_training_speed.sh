#!/bin/bash
# Quick test to validate training speed fix
# This will run training for just 50 steps to verify speed

set -e

# Parse arguments
HF_TOKEN=""
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --wandb-key) WANDB_KEY="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Usage: $0 --hf-token <token> --wandb-key <key>"
  exit 1
fi

# Instance configuration - Small test instance
INSTANCE_TYPE="g5.xlarge"
IMAGE_ID="ami-0c7217cdde317cfec"
KEY_NAME="chave-gpu-nova"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
VOLUME_SIZE=100

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="seriguela-test-speed-${TIMESTAMP}"

echo "============================================"
echo "Launching TEST instance: $INSTANCE_NAME"
echo "Instance type: $INSTANCE_TYPE"
echo "Purpose: Validate training speed (50 steps)"
echo "============================================"

# Create user data script
TEMP_DIR="C:/Users/madeinweb/AppData/Local/Temp"
mkdir -p "$TEMP_DIR" 2>/dev/null
cat > "$TEMP_DIR/userdata_test.sh" <<'EOF'
#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

sleep 5

# Install dependencies
apt-get update
apt-get install -y python3-pip git

# Clone repository
cd /home/ubuntu
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
git checkout experiment/ppo-symbolic-regression

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Setup credentials
echo "huggingface = ${HF_TOKEN}" > /home/ubuntu/.tokens.txt
echo "wandb = ${WANDB_KEY}" >> /home/ubuntu/.tokens.txt
chmod 600 /home/ubuntu/.tokens.txt

# Login to services
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_KEY}"
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_KEY

# Create test script that runs only 50 steps
cat > test_speed.py <<'PYTHON_EOF'
import sys
import time
sys.path.insert(0, '/home/ubuntu/seriguela')

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json

print("="*80)
print("SPEED TEST - Training for 50 steps only")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Add LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

print(f"Model loaded with LoRA")

# Load dataset with EXISTING SPLITS
print("Loading dataset with FIXED method (no double-split)...")
dataset = load_dataset("augustocsc/sintetico_natural_prefix_682k")

print(f"Train size: {len(dataset['train']):,}")
print(f"Validation size: {len(dataset['validation']):,}")

# Convert to JSON format
def convert_to_json_format(example):
    text = example['p_prompt_n_converted']
    lines = text.strip().split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'vars':
                data['vars'] = [v.strip() for v in value.split(',')]
            elif key == 'oper':
                data['ops'] = [o.strip() for o in value.split(',')]
            elif key == 'cons':
                data['cons'] = value
            elif key == 'expr':
                data['expr'] = value
    return {'text': json.dumps(data, ensure_ascii=False)}

# Take small sample for speed test
train_sample = dataset["train"].select(range(1000))
val_sample = dataset["validation"].select(range(200))

train_dataset = train_sample.map(convert_to_json_format, remove_columns=['p_prompt_n_converted'])
eval_dataset = val_sample.map(convert_to_json_format, remove_columns=['p_prompt_n_converted'])

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding=False)

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args - just 50 steps
training_args = TrainingArguments(
    output_dir="./test_output",
    max_steps=50,  # Only 50 steps for speed test
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=10,
    logging_steps=1,
    eval_steps=25,
    save_steps=50,
    eval_strategy="steps",
    fp16=True,
    report_to="none",  # Don't log to wandb for test
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
)

print("\nStarting speed test (50 steps)...")
print("Expected: ~2-3 seconds per step")
print("If > 5s per step, something is wrong!")
print()

start_time = time.time()
trainer.train()
end_time = time.time()

total_time = end_time - start_time
time_per_step = total_time / 50

print("\n" + "="*80)
print("SPEED TEST RESULTS")
print("="*80)
print(f"Total time: {total_time:.1f} seconds")
print(f"Time per step: {time_per_step:.2f} seconds")
print(f"Steps per second: {1/time_per_step:.3f}")
print()

if time_per_step < 4:
    print("✓ SUCCESS: Training speed is GOOD (< 4s/step)")
    print("✓ Fix validated - ready for full training")
elif time_per_step < 8:
    print("⚠ WARNING: Training speed is ACCEPTABLE but slower than expected")
    print("  Consider investigating further")
else:
    print("✗ FAILED: Training speed is TOO SLOW (> 8s/step)")
    print("  Double-split bug may still be present")

print("="*80)
PYTHON_EOF

# Run test
python3 test_speed.py > /home/ubuntu/test_speed.log 2>&1

# Mark completion
touch /home/ubuntu/.test_complete
echo "Test complete at $(date)" >> /home/ubuntu/.test_complete

EOF

# Replace tokens in user data
sed -i "s|\${HF_TOKEN}|${HF_TOKEN}|g" $TEMP_DIR/userdata_test.sh
sed -i "s|\${WANDB_KEY}|${WANDB_KEY}|g" $TEMP_DIR/userdata_test.sh

# Launch instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $IMAGE_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]" \
  --user-data file://$TEMP_DIR/userdata_test.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "============================================"
echo "TEST Instance ready!"
echo "============================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Waiting for setup to complete (this will take ~5 minutes)..."
sleep 300
echo ""
echo "Checking test results:"
echo "  ssh -i ~/.ssh/chave-gpu-nova.pem ubuntu@${PUBLIC_IP}"
echo "  cat ~/test_speed.log"
echo ""
echo "To monitor progress:"
echo "  ssh ubuntu@${PUBLIC_IP} 'tail -f ~/test_speed.log'"
echo ""
echo "To check if test completed:"
echo "  ssh ubuntu@${PUBLIC_IP} 'test -f ~/.test_complete && echo DONE || echo Running'"
echo ""
echo "After validation, STOP this test instance:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "============================================"

rm $TEMP_DIR/userdata_test.sh
