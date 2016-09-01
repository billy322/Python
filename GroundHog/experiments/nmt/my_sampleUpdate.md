encdec.py:
Class RecurrentLayerWithSearch:
update step_fprop: return_alignment=false > true

class Decoder: 
update compute_alignment: false > true, self.compute_alignment = True
add build_decoder_alignment, build_next_probs_predictor_align

class RNNencodeDecoder:
update create_next_probs_computer: self.next_probs_fn include self.decoder.build_next_probs_predictor_align 
update parse_input: add global return: return seq, seqin

sample.py
add header: # -*- coding: utf-8 -*- 
