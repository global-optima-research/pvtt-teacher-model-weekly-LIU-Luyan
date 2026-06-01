import json

with open('/data/liuluyan/davis2017_eval_output/eval_detailed.json') as f:
    data = json.load(f)

key_videos = ['gold-fish', 'lab-coat', 'kite-surf', 'paragliding-launch',
              'india', 'dogs-jump', 'bike-packing', 'bmx-trees']

header = f"{'Video':<22} {'Model':<8} {'J':>6} {'F':>6} {'J&F':>6} {'TC':>6}"
print(header)
print('-' * 60)
for v in key_videos:
    for r in data['per_video']:
        if r['video'] == v:
            line = f"{v:<22} {r['model']:<8} {r['J']:>6.1f} {r['F']:>6.1f} {r['J&F']:>6.1f} {r['temporal_consistency']:>6.1f}"
            print(line)
    print()
