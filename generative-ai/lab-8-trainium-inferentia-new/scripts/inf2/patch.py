import json
path = '/tmp/inf2-compile/compiled/neuron_config.json'
with open(path) as f:
    cfg = json.load(f)
cfg['optimum_neuron_version'] = '0.4.5'
with open(path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patched to 0.4.5')
