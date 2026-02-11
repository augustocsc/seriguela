#!/usr/bin/env python3
from transformers import AutoModelForCausalLM
import sys

try:
    model = AutoModelForCausalLM.from_pretrained('augustocsc/Se124M_700K_infix_v3_json', trust_remote_code=True)
    print('Model exists on HuggingFace')
except Exception as e:
    print(f'Model not found or error: {e}')