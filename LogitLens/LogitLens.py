from LogitLens.LogitVis import LogitVis, Metric
from LogitLens.LMWrapper import wrapModel
from nnsight import LanguageModel
import torch
from functools import lru_cache
from sklearn.decomposition import PCA
import numpy as np

class LensManager():
    
    def __init__(self):
        self.models = {}

    def runLens(self, model : LanguageModel, 
                        prompt : str,
                        metric: Metric,
                        remote : bool = True,
                        heatmap : bool = True,
                        token : int = None):
        """_summary_

        Args:
            model (LanguageModel): nnsight language model
            prompt (str): string prompt
            metric (Metric): Logit lens analysis metric
            remote (bool, optional) : if True, uses nnisght remote api. Defaults to True.
            heatmap (bool, optional): if True, returned figure will be a heatmap. if False, returned figure will be a line graph for a single token position. Defaults to True.
            token (int, optional): Token position if heatmap is False. If no token position is given, defaults to the last token. Defaults to None.
        """
        
        # Initialize logit lens
        key = (model, prompt, remote)
        if key in self.models:
            lens = self.models[key]
        else:
            lens = _LogitLens(model, prompt, remote)
            self.models[key] = lens
        
        # Run
        return lens._explorePrompt(metric, heatmap, token)

class _LogitLens():
    
    """ Runs logit lens analysis """
    
    def __init__(self, model : LanguageModel, prompt : str, remote : bool):
        """
        Prepares to use logit lens on the given model for the given prompt

        Args:
            model (LanguageModel) : Wrapped model
            prompt (str) : prompt as a string
            remote (bool) : if True, use nnsight remote api
        """
        
        self.model = wrapModel(model)
        self.prompt = prompt
        self.remote = remote
        self.max_probs = None
        self.words = None
        self.tokens = None
        self.input_words = None
        self.probs = None
        self.acts = None
        
    def _explorePrompt(self, metric : Metric, heatmap : bool = True, token : int = None):
        """_summary_

        Args:
            metric (Metric): _description_
            heatmap (bool, optional): _description_. Defaults to True.
            token (int, optional): _description_. Defaults to None.
            
        Returns: a figure
        """
        
        #TODO: Check that heatmap and token aren't both true
        
        # Run logit lens
        if self.max_probs is None:
            print("Running logit lens")
            # They all get set at the same time so just checking one
            max_probs, words, tokens, input_words, probs, acts = self._logitLens()
            self.max_probs = max_probs
            self.words = words
            self.tokens = tokens
            self.input_words = input_words
            self.probs = probs
            self.acts = acts
        
        # Get data based on metric
        if metric == Metric.PROB:
            # No additional calculations needed for probability
            data = self.max_probs.detach().cpu().to(torch.float32).numpy().T
        elif metric == Metric.RK:
            ranks = self._rank()
            data = list(map(list, zip(*ranks)))
        elif metric == Metric.KLD:
            kld = self._kld()
            data = kld.detach().cpu().to(torch.float32).numpy().T
        elif metric == Metric.ENT:
            ent = self._entropy()
            data = ent.detach().cpu().to(torch.float32).numpy().T
        elif metric == Metric.PC:
            pcs = self._oneComponentPC()
            data = pcs.T
            # return data
        else:
            raise TypeError("Metric not found")
        
        # Generate the correct figure
        vis = LogitVis(metric, data, self.input_words, self.words)
        if heatmap:
            return vis.heatmap()
        elif metric == Metric.PC:
            return vis.lineplotLayer(token if token else -1)
        else:
            return vis.lineplotToken(token if token else -1)
    
    @lru_cache()
    def _logitLens(self):
        """
        Runs logit lens
        
        Returns:
            max_probs (torch.Tensor) : Maximum probabilities for each token
            words (List[List[str]]) : Token predictions for each layer
            tokens (torch.tensor) : Encoded token predictions for each layer
            input_words (List[str]) : List of input words (tokenized prompt)
            probs (torch.Tensor) : Probabilities for each token
            acts (torch.Tensor) : Activations for each layer
        """
        
        # Save probabilities by layer
        probs_layers = []
        
        # Store activations
        acts = []
        
        with self.model.trace(remote=self.remote) as tracer:
            
            with tracer.invoke(self.prompt) as invoker:
                
                for layer_idx, layer in enumerate(self.model.layers):
                    
                    lnf = self.model.ln_f(layer.output[0])
                    layer_output = self.model.lm_head(lnf)
                    
                    acts.append(layer_output.cpu().save())
                    
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)
        
        probs = torch.cat([probs.value for probs in probs_layers])
        acts = torch.cat(acts).to(torch.float32)
        # Find the maximum probability and corresponding tokens for each position
        max_probs, tokens = probs.max(dim=-1)
        # Decode token IDs to words for each layer
        words = [
            [self.model.tokenizer.decode(t.cpu()).encode("utf-8").decode() for t in layer_tokens]
            for layer_tokens in tokens]
        # Access the 'input_ids' attribute of the invoker object to get the input words
        input_words = [self.model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
        
        return max_probs, words, tokens, input_words, probs, acts
        
    def _rank(self):
        """
        Ranks tokens
        """
        finalProbs = self.probs[-1]
        
        # Sort the final probabilities in descending order
        sortedIndices = torch.argsort(finalProbs, descending=True)
        
        # Map idx to rank
        rankMap = [{idx.item(): rank for rank, idx in enumerate(_)} for _ in sortedIndices]
        
        # Determine ranks for each token id
        ranks = [
            [rankMap[idx][word_id.item()] for idx, word_id in enumerate(layer)]
            for layer in self.tokens
        ]
        
        return ranks
    
    def _entropy(self):
        """
        Calculates entropy
        """
        
        return -torch.sum(self.probs * torch.log(self.probs), dim=-1)
    
    def _kld(self):
        """
        Calculates KL Divergence 
        """
        
        return torch.sum(self.probs * torch.log(self.probs / self.probs[-1]), dim=-1)
    
    def _oneComponentPC(self):
        """
        Calculates 1 component PCA for each layer
        """

        num_layers = self.acts.shape[0]
        layer_pcas = []
        
        for layer in range(num_layers):
            activations = self.acts[layer, :, :]
            
            pca = PCA(n_components=1)
            layer_pca = pca.fit_transform(activations)
            
            layer_pcas.append(layer_pca)
            
        return np.squeeze(layer_pcas)
        
    
        