#!/usr/bin/env python3
"""
Check which configs are missing from completed runs.

Usage:
    # Check specific instance via SSH
    python check_missing_configs.py --ssh ubuntu@IP_ADDRESS

    # Check local wandb directory
    python check_missing_configs.py --local /path/to/wandb

    # Check all 6 instances (requires AWS and instances running)
    python check_missing_configs.py --check-all-aws
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter, defaultdict


def load_expected_configs(remaining_file="remaining_base_configs.json"):
    """Load expected configs from remaining_base_configs.json"""
    with open(remaining_file) as f:
        return json.load(f)


def parse_wandb_metadata(metadata_file):
    """Extract config tuple from wandb-metadata.json"""
    try:
        with open(metadata_file) as f:
            meta = json.load(f)

        args = meta.get('args', [])
        config = {}

        # Parse command line args
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                key = arg[2:]
                if i+1 < len(args) and not args[i+1].startswith('--'):
                    config[key] = args[i+1]

        # Extract model (remove augustocsc/ prefix and _682k suffix)
        model = config.get('model', '').split('/')[-1]
        model = model.replace('gpt2_', '').replace('_682k', '')

        problem = config.get('problem', '')
        algo = config.get('algorithm', '')
        reward = config.get('reward', '')
        penalty = config.get('penalty', '')
        temp = config.get('temperature', '')
        prompt = config.get('prompt_type', '')

        # Get noise level
        noise_level = 0.0
        if '--noise_level' in args:
            idx = args.index('--noise_level')
            if idx + 1 < len(args):
                try:
                    noise_level = float(args[idx + 1])
                except:
                    pass

        return [model, problem, algo, reward, penalty, temp, prompt, noise_level]

    except Exception as e:
        return None


def scan_wandb_directory(wandb_dir):
    """Scan wandb directory and extract all completed configs"""
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        raise FileNotFoundError(f"wandb directory not found: {wandb_dir}")

    run_dirs = [d for d in wandb_path.iterdir() if d.name.startswith('run-')]
    print(f"Found {len(run_dirs)} run directories")

    completed_configs = []
    errors = 0

    for run_dir in run_dirs:
        meta_file = run_dir / 'files' / 'wandb-metadata.json'
        if meta_file.exists():
            config = parse_wandb_metadata(meta_file)
            if config:
                completed_configs.append(config)
            else:
                errors += 1

    print(f"Parsed {len(completed_configs)} configs ({errors} errors)")
    return completed_configs


def scan_ssh_wandb(ssh_target, key_file=None):
    """Scan wandb directory on remote instance via SSH"""
    print(f"Connecting to {ssh_target}...")

    # Build SSH command
    ssh_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no']
    if key_file:
        ssh_cmd.extend(['-i', key_file])
    ssh_cmd.append(ssh_target)

    # Python script to run on remote
    remote_script = """
import os
import json

wandb_dir = '~/seriguela/2_training/reinforcement/wandb'
wandb_dir = os.path.expanduser(wandb_dir)

run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith('run-')]
print(f'RUNS:{len(run_dirs)}')

completed = []

for run_dir in run_dirs:
    meta_file = os.path.join(wandb_dir, run_dir, 'files', 'wandb-metadata.json')
    if os.path.exists(meta_file):
        try:
            with open(meta_file) as f:
                meta = json.load(f)

            args = meta.get('args', [])
            config = {}

            for i, arg in enumerate(args):
                if arg.startswith('--'):
                    key = arg[2:]
                    if i+1 < len(args) and not args[i+1].startswith('--'):
                        config[key] = args[i+1]

            model = config.get('model', '').split('/')[-1]
            model = model.replace('gpt2_', '').replace('_682k', '')
            problem = config.get('problem', '')
            algo = config.get('algorithm', '')
            reward = config.get('reward', '')
            penalty = config.get('penalty', '')
            temp = config.get('temperature', '')
            prompt = config.get('prompt_type', '')

            noise_level = 0.0
            if '--noise_level' in args:
                idx = args.index('--noise_level')
                if idx + 1 < len(args):
                    noise_level = float(args[idx + 1])

            completed.append([model, problem, algo, reward, penalty, temp, prompt, noise_level])
        except:
            pass

print('CONFIGS:' + json.dumps(completed))
"""

    ssh_cmd.append(f'python3 -c "{remote_script}"')

    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)

    # Parse output
    for line in result.stdout.split('\n'):
        if line.startswith('RUNS:'):
            runs = int(line.split(':')[1])
            print(f"Found {runs} runs on remote")
        elif line.startswith('CONFIGS:'):
            configs_json = line.split('CONFIGS:')[1]
            configs = json.loads(configs_json)
            print(f"Parsed {len(configs)} configs")
            return configs

    raise RuntimeError(f"Failed to parse SSH output: {result.stdout}\n{result.stderr}")


def compare_configs(expected, completed):
    """Compare expected vs completed configs and find missing"""
    expected_set = set(tuple(c) for c in expected)
    completed_set = set(tuple(c) for c in completed)

    missing = expected_set - completed_set
    duplicates = len(completed) - len(completed_set)

    return {
        'expected': len(expected_set),
        'completed': len(completed_set),
        'missing': sorted([list(m) for m in missing]),
        'duplicates': duplicates,
        'coverage': len(completed_set) / len(expected_set) * 100 if expected_set else 0
    }


def analyze_missing_patterns(missing_configs):
    """Analyze patterns in missing configs"""
    if not missing_configs:
        return {}

    patterns = {
        'models': Counter(),
        'problems': Counter(),
        'algorithms': Counter(),
        'rewards': Counter(),
        'penalties': Counter(),
        'temperatures': Counter(),
        'prompts': Counter(),
        'noise_levels': Counter(),
    }

    for c in missing_configs:
        patterns['models'][c[0]] += 1
        patterns['problems'][c[1]] += 1
        patterns['algorithms'][c[2]] += 1
        patterns['rewards'][c[3]] += 1
        patterns['penalties'][c[4]] += 1
        patterns['temperatures'][c[5]] += 1
        patterns['prompts'][c[6]] += 1
        patterns['noise_levels'][c[7]] += 1

    return patterns


def main():
    parser = argparse.ArgumentParser(description="Check missing configs from Phase A")
    parser.add_argument('--ssh', type=str, help='SSH target (e.g., ubuntu@IP)')
    parser.add_argument('--key', type=str, default='C:/Users/madeinweb/chave-gpu.pem',
                       help='SSH key file')
    parser.add_argument('--local', type=str, help='Local wandb directory path')
    parser.add_argument('--check-all-aws', action='store_true',
                       help='Check all 6 AWS instances (requires instances running)')
    parser.add_argument('--remaining-file', type=str,
                       default='remaining_base_configs.json',
                       help='Path to remaining_base_configs.json')
    parser.add_argument('--output', type=str,
                       help='Save missing configs to JSON file')

    args = parser.parse_args()

    # Load expected configs
    print("Loading expected configs...")
    all_expected = load_expected_configs(args.remaining_file)

    # Get completed configs
    completed_configs = []

    if args.ssh:
        completed_configs = scan_ssh_wandb(args.ssh, args.key)
    elif args.local:
        completed_configs = scan_wandb_directory(args.local)
    elif args.check_all_aws:
        print("AWS checking not implemented yet. Use --ssh for each instance.")
        return
    else:
        parser.print_help()
        return

    # Determine model-problem from completed configs
    if completed_configs:
        model_problem = f"{completed_configs[0][0]}_{completed_configs[0][1]}"
        print(f"\nDetected model-problem: {model_problem}")

        # Filter expected configs for this model-problem
        expected_filtered = [c for c in all_expected
                           if c[0] == completed_configs[0][0]
                           and c[1] == completed_configs[0][1]]
    else:
        expected_filtered = all_expected
        model_problem = "unknown"

    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    results = compare_configs(expected_filtered, completed_configs)

    print(f"Expected configs:  {results['expected']}")
    print(f"Completed configs: {results['completed']}")
    print(f"Missing configs:   {len(results['missing'])}")
    print(f"Duplicate runs:    {results['duplicates']}")
    print(f"Coverage:          {results['coverage']:.2f}%")

    if results['missing']:
        print(f"\n{len(results['missing'])} MISSING CONFIGS:")
        print("-" * 60)

        # Analyze patterns
        patterns = analyze_missing_patterns(results['missing'])

        print("\nMissing by pattern:")
        for key, counter in patterns.items():
            if counter:
                print(f"  {key}:")
                for value, count in counter.most_common():
                    print(f"    {value}: {count}")

        print("\nFirst 20 missing configs:")
        for i, config in enumerate(results['missing'][:20], 1):
            print(f"  {i}. {config}")

        if len(results['missing']) > 20:
            print(f"  ... and {len(results['missing']) - 20} more")

        # Save to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results['missing'], f, indent=2)
            print(f"\nSaved missing configs to: {args.output}")
    else:
        print("\n✓ All configs completed!")


if __name__ == '__main__':
    main()
