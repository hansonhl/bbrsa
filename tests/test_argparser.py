from onmt.utils.parse import ArgumentParser
import onmt.opts as opts

parser = ArgumentParser()
opts.config_opts(parser)
opts.translate_opts(parser)

optstring = ['-batch_size', '20', '-beam_size', '10', '-model',  '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt', '-src', '/home/hansonlu/myOpenNMT/data/giga/small_input.txt', '-seed', '-1']

opt = parser.parse_args(optstring)

print(opt)
