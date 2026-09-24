"""Serve the real-clip walkthrough on loopback only, without publishing dataset assets."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path.home() / '.cache/auslan-walkthrough')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    if not (args.assets / 'manifest.json').is_file():
        parser.error('Run demo/export_walkthrough.py first.')
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from demo.app import Settings, create_app
    app = create_app(Settings(mode='showcase'))
    app.mount('/research-example', StaticFiles(directory=args.assets), name='local-research-example')
    print(f'Open http://127.0.0.1:{args.port}/?example=local')
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)
