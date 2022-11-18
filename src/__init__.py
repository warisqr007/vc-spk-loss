from .mel_decoder_mol_encAddlf0 import MelDecoderMOL
from .mel_decoder_mol_v2 import MelDecoderMOLv2
from .rnn_ppg2mel import BiRnnPpg2MelModel
from .mel_decoder_lsa import MelDecoderLSA
from .transformer_bnftomel import Transformer

def build_model(model_name: str):
    if model_name == "transformer-vc":
        return Transformer
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
