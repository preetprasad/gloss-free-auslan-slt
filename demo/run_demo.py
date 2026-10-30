#!/usr/bin/env python3
"""Run the research showcase and the live Auslan-to-English translator locally."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--showcase', action='store_true', help='Pages only; no camera translation')
    parser.add_argument('--preview', action='store_true', help='Explicit UI sample mode; never enabled in production')
    parser.add_argument('--models', default='single,ensemble', help='Operating points to load: single, ensemble or both')
    parser.add_argument('--coverage', type=float, default=0.5, choices=[0.5, 0.75, 0.9],
                        help='Target answer rate; each model uses its validation-fitted threshold for this rate')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--rgb-device', default='cpu', choices=['cpu', 'mps', 'cuda'], help='DINOv2 device for the ensemble')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.preview and args.showcase: parser.error('Use --showcase or --preview, not both.')
    os.environ['DEMO_MODE'] = 'sample' if args.preview else 'showcase' if args.showcase else 'model'
    os.environ['DEMO_MODELS'] = args.models
    os.environ['DEMO_COVERAGE'] = str(args.coverage)
    os.environ['DEMO_DEVICE'] = args.device
    os.environ['DEMO_RGB_DEVICE'] = args.rgb_device
    import uvicorn
    uvicorn.run('demo.app:app', host=args.host, port=args.port, workers=1,
                ws_max_size=1_350_000, ws_max_queue=2, timeout_keep_alive=10, access_log=False)

if __name__ == '__main__': main()
