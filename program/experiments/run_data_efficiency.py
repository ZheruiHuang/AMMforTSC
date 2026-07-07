import argparse
import subprocess
import sys
from pathlib import Path


def parse_budget_list(raw):
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_config', type=str, required=True)
    parser.add_argument('--dynamics_model_dir', type=str, required=True)
    parser.add_argument('--adapter_model_dir', type=str, default='')
    parser.add_argument('--full_budget_episodes', type=int, default=60)
    parser.add_argument('--budgets', type=str, default='0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--thread_num', type=int, default=20)
    parser.add_argument('--max_step', type=int, default=180)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='outputs/data_efficiency')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    target_script = repo_root / 'program' / 'train' / 'target_adapt.py'
    budgets = parse_budget_list(args.budgets)

    for budget in budgets:
        episodes = max(1, round(args.full_budget_episodes * budget))
        for run_idx in range(args.runs):
            save_dir = Path(args.output_dir) / f'budget_{budget:.2f}' / f'run_{run_idx}'
            command = [
                sys.executable,
                str(target_script),
                '--config',
                args.target_config,
                '--dynamics_model_dir',
                args.dynamics_model_dir,
                '--episode',
                str(episodes),
                '--save_model_freq',
                str(episodes),
                '--save_model_dir',
                str(save_dir),
                '--thread_num',
                str(args.thread_num),
                '--max_step',
                str(args.max_step),
                '--stride',
                str(args.stride),
            ]
            if args.adapter_model_dir:
                command.extend(['--adapter_model_dir', args.adapter_model_dir])

            print(f'Running target budget={budget:.2f}, episodes={episodes}, run={run_idx}')
            subprocess.run(command, cwd=repo_root, check=True)


if __name__ == '__main__':
    main()
