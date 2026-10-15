import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from kit import Simulator, generate, VERSION

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
LIMIT = 1024 * 1024


def strict_json(text):
    def reject(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(text, parse_constant=reject)


def validate_response(response, phase):
    key = 'asks' if phase == 'ask' else 'pairs'
    if not isinstance(response, dict) or set(response) != {key, 'memory'}:
        raise ValueError('Response must contain exactly ' + key + ' and memory')
    if not isinstance(response[key], list):
        raise ValueError(key + ' must be an array')
    if len(json.dumps(response['memory'], allow_nan=False).encode()) > LIMIT:
        raise ValueError('Memory exceeds 1 MiB')
    if key == 'asks':
        for ask in response[key]:
            if not isinstance(ask, dict) or set(ask) != {'member_id', 'field'} or not all(isinstance(v, str) for v in ask.values()):
                raise ValueError('Invalid ask shape')
    else:
        for pair in response[key]:
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(i, str) for i in pair):
                raise ValueError('Each pair must be an array of two strings')
    return response


def run_phase(proc, phase, observation, memory):
    payload = json.dumps({'phase': phase, 'observation': observation, 'memory': memory}, allow_nan=False) + '\n'
    proc.stdin.write(payload.encode('utf-8'))
    proc.stdin.flush()
    start = time.perf_counter()
    line = proc.stdout.readline()
    elapsed = time.perf_counter() - start
    if not line:
        raise ValueError('Policy process closed stdout unexpectedly')
    response = validate_response(strict_json(line.decode('utf-8')), phase)
    return response[phase if phase == 'ask' else 'pairs'], response['memory'], elapsed



def docker_command(image_tag):
    return ['docker', 'run', '--rm', '-i', '--name', 'sequential-' + uuid.uuid4().hex, '--network=none', '--cpus=2',
            '--memory=4g', '--pids-limit=256', '--read-only', image_tag]



def episode(world, command, decision_days=60, follow_up_days=40):
    simulator = Simulator(world)
    memory = None
    inference_seconds = 0.0
    action_hashes = []
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(decision_days):
            observation = simulator.observe()
            asks, memory, elapsed_ask = run_phase(proc, 'ask', observation, memory)
            inference_seconds += elapsed_ask
            action_hashes.append(hashlib.sha256(json.dumps(asks, sort_keys=True).encode()).digest())
            simulator.resolve_asks(asks)

            observation = simulator.observe()
            pairs, memory, elapsed_pair = run_phase(proc, 'pair', observation, memory)
            inference_seconds += elapsed_pair
            action_hashes.append(hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).digest())
            simulator.advance(pairs)

        for _ in range(follow_up_days):
            simulator.advance([])
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    metrics = simulator.metrics()
    digest = hashlib.sha256()
    for item in action_hashes:
        digest.update(item)

    result = {'valid': True, 'seed': world['seed'], 'variant': world['variant'], 'schema_version': VERSION}
    result.update(metrics)
    result.update(inference_seconds=inference_seconds, action_sha256=digest.hexdigest())
    return result




def summarise(rows):
    if not rows or any(not r.get('valid') for r in rows):
        return {'eligible': False, 'primary_score': None, 'reason': 'At least one episode is invalid'}
    families = collections.defaultdict(list)
    for row in rows:
        families[row['variant']].append(row)
    keys = ('msmi_per_100_arrived_members', 'coverage', 'mutual_acceptances_per_100', 'ask_cost', 'inference_seconds')
    family_means = {name: {key: statistics.mean(r[key] for r in rs) for key in keys}
                    for name, rs in sorted(families.items())}

    overall = {key: statistics.mean(f[key] for f in family_means.values()) for key in keys}
    return {'eligible': True, 'primary_score': overall['msmi_per_100_arrived_members'],
            'scenario_means': family_means, 'overall': overall,
            'ranking_key_descending': [overall['msmi_per_100_arrived_members'], overall['coverage'],
                                       overall['mutual_acceptances_per_100'], -overall['ask_cost'], -overall['inference_seconds']]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='policy.py')
    p.add_argument('--baseline', choices=['febf', 'febf_scarcity', 'greedy', 'simple_fifo', 'no_asks', 'random'], default='febf')
    p.add_argument('--image', help='Use isolated offline Docker execution instead of trusted local subprocesses')
    p.add_argument('--seeds', default='101', help='Comma-separated public experiment seeds')
    p.add_argument('--variants', default='development', help='Comma-separated variants, or all')
    p.add_argument('--output', default='results/evaluation.json')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    variants = list(VARIANTS) if args.variants == 'all' else args.variants.split(',')
    if not seeds or not variants or any(v not in VARIANTS for v in variants):
        p.error('Choose at least one seed and recognised variant')
    command = docker_command(args.image) if args.image else [sys.executable, args.policy, '--baseline', args.baseline]
    rows = []
    for variant in variants:
        for seed in seeds:
            try:
                row = episode(generate(seed, 200, 'evaluation', variant), command)
            except (ValueError, TypeError, KeyError, OSError) as error:
                row = {'valid': False, 'error': str(error)}
            row.update(seed=seed, variant=variant)
            rows.append(row)
            print(json.dumps({'seed': seed, 'variant': variant, 'valid': row['valid'],
                              'msmi': row.get('mutual_second_meeting_intention')}), flush=True)
    summary = summarise(rows)
    output = {'release': VERSION, 'mode': 'container' if args.image else 'trusted_local',
              'decision_days': 60, 'follow_up_days': 40, 'members_per_episode': 200,
              'episodes': rows, 'summary': summary}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
