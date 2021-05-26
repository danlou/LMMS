import sys
from wn_utils import WN_Utils

wn_utils = WN_Utils()

model_outputs_path = sys.argv[1]

if '.key.' in model_outputs_path:
    model_outputs_syn_path = model_outputs_path.replace('.key.', '.syn.key.')
elif model_outputs_path.endswith('.key'):
    model_outputs_syn_path = model_outputs_path.replace('.key', '.syn.key')


print('Writing %s ...' % model_outputs_syn_path)
with open(model_outputs_syn_path, 'w') as outputs_converted_f:
    with open(model_outputs_path) as outputs_f:
        for line in outputs_f:
            elems = line.strip().split()
            inst_id, inst_sks = elems[0], elems[1:]

            inst_syns = [wn_utils.sk2syn(sk).name() for sk in inst_sks]

            outputs_converted_f.write('%s %s\n' % (inst_id, ' '.join(list(set(inst_syns)))))
