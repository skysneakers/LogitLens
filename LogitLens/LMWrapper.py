"""
Simple API for working with logit lens 
"""

from nnsight import LanguageModel

def wrapModel(model: LanguageModel):
    """ Check if the language model is ll or gpt2, raise NYI error if not
    Wraps and returns model

    Args:
        model (LanguageModel): Unwrapped language model
    
    Returns: wrapped model
    """
    ## TODO: Check model type
    
    return LMWrapper._create_llama8b_wrapper(model)

class LMWrapper():
    
    """
    A wrapper class for language models
    """
    
    def __init__(self, model: LanguageModel, layer_dict: dict):
        self.model = model
        self.trace = model.trace
        
        # For each of the layers in the model, get the layer functions
        self.layers = layer_dict.get('layers')
        self.attention_fn = layer_dict.get('attention_fn')
        self.mlp_fn = layer_dict.get('mlp_fn')
        self.ln_f = layer_dict.get('ln_f')
        self.lm_head = layer_dict.get('lm_head')
        self.tokenizer = layer_dict.get('tokenizer')
        self.last_layer = layer_dict.get('last_layer')
    
    # Pre-wrapped models
    
    def _create_gpt2_wrapper(gpt2_model):
        return LMWrapper(
            model=gpt2_model,
            layer_dict={
                'layers': gpt2_model.transformer.h,
                'attention_fn': lambda layer: layer.attn(layer.ln_1(layer.input)),
                'mlp_fn': lambda layer, attn_output: layer.mlp(layer.ln_2(attn_output)),
                'ln_f': gpt2_model.transformer.ln_f,
                'lm_head': gpt2_model.lm_head,
                'tokenizer': gpt2_model.tokenizer,
                'last_layer': gpt2_model.transformer.h[-1]
            }
        )
    
    def _create_llama8b_wrapper(llama_model):
        return LMWrapper(
            model = llama_model,
            layer_dict={
                'layers': llama_model.model.layers,
                'attention_fn': lambda layer: layer.attention(layer.norm1(layer.input)),
                'mlp_fn': lambda layer, attn_output: layer.mlp(layer.norm2(attn_output)),
                'ln_f': llama_model.model.norm,
                'lm_head': llama_model.lm_head,
                'tokenizer': llama_model.tokenizer,
                'last_layer': llama_model.model.layers[-1]
            }
        )

