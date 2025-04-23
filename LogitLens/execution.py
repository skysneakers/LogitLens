"""
Functions for exploring each logit lens metric
Caches calls to functions
"""

from LogitLens.LogitLens import LensManager
from LogitLens.LogitVis import Metric
from nnsight import LanguageModel
from functools import lru_cache

_manager = LensManager()

@lru_cache()
def logitProbability(model : LanguageModel, 
                     prompt : str,
                     remote : bool = True,
                     heatmap : bool = True,
                     token : int = None):
    """
    Run the logit lens to explore probability

    Args:
        model (LanguageModel): nnsight language model
        prompt (str): string prompt
        remote (bool, optional) : if True, uses nnisght remote api. Defaults to True.
        heatmap (bool, optional): if True, returned figure will be a heatmap. if False, returned figure will be a line graph for a single token position. Defaults to True.
        token (int, optional): Token position if heatmap is False. If no token position is given, defaults to the last token. Defaults to None.
    """

    return _manager.runLens(model, prompt, Metric.PROB, remote, heatmap, token)
    
    
@lru_cache()
def logitKLDivergence(model : LanguageModel, 
                     prompt : str,
                     remote : bool = True,
                     heatmap : bool = True,
                     token : int = None):
    return _manager.runLens(model, prompt, Metric.KLD, remote, heatmap, token)

@lru_cache() 
def logitPC(model : LanguageModel, 
                     prompt : str,
                     remote : bool = True,
                     heatmap : bool = True,
                     layer : int = None):
    """
    Run the logit lens to explore single-components

    Args:
        model (LanguageModel): nnsight language model
        prompt (str): string prompt
        remote (bool, optional) : if True, uses nnisght remote api. Defaults to True.
        heatmap (bool, optional): if True, returned figure will be a heatmap. if False, returned figure will be a line graph for a single token position. Defaults to True.
        layer (int, optional): Layer position if heatmap is False. If no layer position is given, defaults to the last layer. Defaults to None.
    """
    return _manager.runLens(model, prompt, Metric.PC, remote, heatmap, layer)

@lru_cache()    
def logitEntropy(model : LanguageModel, 
                     prompt : str,
                     remote : bool = True,
                     heatmap : bool = True,
                     token : int = None):
    return _manager.runLens(model, prompt, Metric.ENT, remote, heatmap, token)
    
@lru_cache()
def logitRank(model : LanguageModel, 
                     prompt : str,
                     remote : bool = True,
                     heatmap : bool = True,
                     token : int = None):
    
    return _manager.runLens(model, prompt, Metric.RK, remote, heatmap, token)
    
# @lru_cache()
# def logitRaw():