import sys, os, json, pickle, math, random
import numpy as np, pandas as pd
from tqdm import tqdm
import itertools
from functools import wraps, partial
from tqdm import trange

import torch
from torch import nn
import torch.nn.functional as F

import datasets
from transformers import LlamaForCausalLM, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)

import emoji
from retrieval_head_detection import SentenceSampler
from typing import Optional, Union, List, Dict, Tuple
from loguru import logger
logger.info(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auto_read_data(file_path, return_format="list", print_log=False):
    """
    Read data from a file and return it in the specified format.

    Parameters:
        file_path (str): The path to the file to be read.
        return_format (str, optional): The format in which the data should be returned. Defaults to "list".

    Returns:
        list or str: The data read from the file, in the specified format.
    """

    file_type = file_path.split('.')[-1].lower()  

    # Get the size of the file right after it's been written to
    file_size = os.path.getsize(file_path)
    # Convert the size to a more readable format
    readable_size = convert_size(file_size)
    if print_log:
        logger.info(f"begin to read data from {file_path} | file size: {readable_size} | file type: {file_type}")
    try:
        if file_type == 'jsonl':  
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = [json.loads(line.strip()) for line in file]  
        elif file_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = json.load(file)
        elif file_type == 'pkl':  
            with open(file_path, 'rb') as file:  
                data = pickle.load(file)  
        elif file_type == 'txt':  
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = [line.strip() for line in file]  
        elif file_type == 'csv':
            raw_data = pd.read_csv(file_path)
            data = raw_data.to_dict(orient='records')  # list[Dict]
        else:  
            raise ValueError(f"Unsupported file type: {file_type}")  
    except:
        raise ValueError(
            f"Error reading file: {file_path}, \
            content didn't match the file type {file_type}, \
            check your data format!"
        )
    
    if return_format != "list":  
        raise ValueError(f"Unsupported return format: {return_format}")  
  
    return data  


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def auto_save_data(lst: Optional[List|Dict], file_path):
    """
    Save a list of items to a file.
    Automatically detect the file type by the suffix of the file_path.

    Args:
        lst (List): The list of items to be saved.
        file_path (str): The path to the file.

        //* Support file types
            - jsonl
            - pkl
            - txt
        *//
    
    Attention:
        Input must by in a list, even if there is only one item.
        e.g., auto_save_data([item], file_path)
        
    Raises:
        ValueError: If the file type is not supported.
    """
    
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"{data_dir} not exist! --> Create data dir {data_dir}")
    suffix_ = file_path.split(".")[-1]
    
    if suffix_ == "jsonl":
        with open(file_path, "w") as f:
            for item in lst:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        logger.info("jsonl file saved successfully!")
    
    elif suffix_ == "json":
        with open(file_path, "w") as f:
            json.dump(lst, f, ensure_ascii=False)
        logger.info("json file saved successfully!")

    elif suffix_ == "pkl":
        with open(file_path, "wb") as f:
            pickle.dump(lst, f)
        logger.info("pkl file saved successfully!")
        
    elif suffix_ == "txt":
        with open(file_path, "w") as f:
            for item in lst:
                f.write(item + "\n")
        logger.info("txt file saved successfully!")
    else:
        raise ValueError(f"file_type {suffix_} not supported!")
    
    # Get the size of the file right after it's been written to
    file_size = os.path.getsize(file_path)
    # Convert the size to a more readable format
    readable_size = convert_size(file_size)

    logger.info(f"Save file to {file_path} | len: {len(lst)} |  size: {readable_size}")

def hack_attn_qwen3(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_adapter=None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # attention_interface: Callable = eager_attention_forward
    # if self.config._attn_implementation != "eager":
    #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # attn_output, attn_weights = attention_interface(
    #     self,
    #     query_states,
    #     key_states,
    #     value_states,
    #     attention_mask,
    #     dropout=0.0 if not self.training else self.attention_dropout,
    #     scaling=self.scaling,
    #     sliding_window=self.sliding_window,  # diff with Llama 这有啥用啊?
    #     **kwargs,
    # )
    dropout=0.0 if not self.training else self.attention_dropout

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) # FIXME: use bf16 to save memory

    if attention_adapter is not None:  # pass attention weights to adapter
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def hack_attn_llama(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_adapter=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 适配新旧版本 --------------------- 
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} 
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) # FIXME: use bf16 to save memory
    
    # 主要功能点
    if attention_adapter is not None:  # pass attention weights to adapter
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def hack_forward_qwen3(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    zeroembed_adapter = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    
    # if zeroembed_adapter is not None:
    #     inputs_embeds = zeroembed_adapter(input_ids, inputs_embeds)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if zeroembed_adapter is not None:
            hidden_states = zeroembed_adapter(input_ids, hidden_states)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def hack_forward_llama(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        zeroembed_adapter = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        
        if zeroembed_adapter is not None:
            inputs_embeds = zeroembed_adapter(input_ids, inputs_embeds)
            

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class AttentionAdapter(AttentionAdapterBase):
    def __init__(self) -> None:
        super().__init__()

    def _forward(self, attn_weights: torch.Tensor):
        self.attn_weights = attn_weights
        self.attn_weights.retain_grad()
        return self.attn_weights

    @property
    def grad(self):
        return self.attn_weights.grad

    @property
    def weight(self):
        return self.attn_weights
    
    @property
    def saliency(self):
        return self.attn_weights * self.attn_weights.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.attn_weights.grad is not None:
            if set_to_none:
                self.attn_weights.grad = None
            else:
                self.attn_weights.grad.zero_()

def manager_decoractor(manager):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator



class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, model_name: str, with_adapter: bool = False):
        self.model = model
        self.model_name = model_name
        self.with_adapter = with_adapter
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            if attention_adapter is None:continue
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self,set_to_none=True):
        
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                if attention_adapter is None:continue
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                if attention_adapter is None:continue
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad,use_abs = True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self,*args,**kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            if attention_adapter is None:continue
            grads.append(self.grad_process(attention_adapter.grad,*args,**kwargs))
        return grads
    
    def saliency(self,*args,**kwargs):
        saliencies= []

        for attention_adapter in self.attention_adapters:
            if attention_adapter is None:continue
            saliencies.append(self.grad_process(attention_adapter.saliency,*args,**kwargs))
        return saliencies

    def weight(self, *args, **kwargs):
        weights = []
        for attention_adapter in self.attention_adapters:
            if attention_adapter is None:continue
            weights.append(self.grad_process(attention_adapter.weight,*args,**kwargs))
        return weights



class AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, model_name: str, with_adapter: bool = False, start_layer = 0):
        self.start_layer = start_layer

        super().__init__(model, model_name, with_adapter)
        self.model_name = model_name
        

    def register_attentioner_to_model(self):
        attention_adapters = []
        if self.with_adapter:
            layer_module = self.model.base_model.model.model.layers
        else:
            layer_module = self.model.model.layers
        for i, layer in enumerate(layer_module):
            if i < self.start_layer:
                # print("ignore layer:",i,"!")
                attention_adapters.append(None)
                continue
            attention_adapter = AttentionAdapter()
            if "llama" in self.model_name.lower() or "tulu" in self.model_name.lower():
                layer.self_attn.forward = partial(hack_attn_llama, layer.self_attn, attention_adapter=attention_adapter)
            elif "qwen" in self.model_name.lower():
                layer.self_attn.forward = partial(hack_attn_qwen3, layer.self_attn, attention_adapter=attention_adapter)
            else:
                raise NotImplementedError(f"{self.model_name} not supported")
            attention_adapters.append(attention_adapter)
        return attention_adapters


class ZeroEmbedAdapter(nn.Module):
    def __init__(self, nonzero_poss, factor):
        super().__init__()
        self.nonzero_poss = sorted(set(nonzero_poss))
        self.factor = factor
    
        self.input_embeddings = []

    def forward(self, input_ids, input_embeds):
        self.input_embeddings.append(input_embeds)
        self.input_embeddings[-1].retain_grad()
        return self.input_embeddings[-1]

        # if len(self.input_embeddings) == 0:
        #     self.input_embeddings.append(input_embeds)
        #     self.input_embeddings[-1].retain_grad()
        #     return self.input_embeddings[-1]

        # print("The", len(self.input_embeddings) + 1,"nd")

        # fac_poss = torch.zeros_like(input_embeds) + self.factor * self.input_embeddings[-1].grad
        # for span_idx in self.nonzero_poss:
        #     fac_poss[:, span_idx[0]:span_idx[1]] = 0
            
        # self.input_embeddings.append((self.input_embeddings[-1] - fac_poss).detach())
        # self.input_embeddings[-1].requires_grad = True
        # self.input_embeddings[-1].retain_grad()
        
        # return self.input_embeddings[-1]

    def zero_grad(self):
        self.inputs_embeds.grad.zero_()


class ZeroEmbedManager:
    def __init__(self, model: PreTrainedModel, model_name: str,
                 nonzero_poss,
                 factor 
                 ):
        self.model = model
        self.model_name = model_name
        # self.attention_adapters = self.register_attentioner_to_model()
        self.zeroembed_adapter = ZeroEmbedAdapter(nonzero_poss, factor)
        if "llama" in self.model_name.lower() or "tulu" in self.model_name.lower():
            self.model.model.forward = partial(hack_forward_llama, 
                                           self.model.model, 
                                           zeroembed_adapter = self.zeroembed_adapter)
        elif "qwen" in self.model_name.lower():
            self.model.model.forward = partial(hack_forward_qwen3, 
                                           self.model.model, 
                                           zeroembed_adapter = self.zeroembed_adapter)

        
    
    def zero_grad(self):
        self.zeroembed_adapter.zero_grad()

    def grad_process(self, grad, p = 1):
        assert len(grad.shape) == 3

        return torch.norm(grad, p = p, dim = -1)

    def get_input_grad(self):
        # return self.grad_process(self.zeroembed_adapter.input_embeddings[-1].grad)
        grads = []
        for i in range(len(self.zeroembed_adapter.input_embeddings)):
            grads.append(self.grad_process(self.zeroembed_adapter.input_embeddings[i].grad))
        return grads



def np_topk(arr, k):
    sorted_indices = np.argsort(arr)
    topk_indices = sorted_indices[-k:]
    topk_values = arr[topk_indices]
    return topk_values, topk_indices


def multi_torch_topk(saliency, target_poss, k):
    values = torch.full((saliency.shape[-1],), 0.)
    for target_pos in range(*target_poss):
        topk_values ,topk_indices = np_topk(saliency[target_pos, :], k)
        topk_values = torch.tensor(topk_values).flatten()
        topk_indices = torch.tensor(topk_indices).flatten()
        values[topk_indices] += topk_values
    
    return np_topk(values.numpy(), k)


def cal_temp(saliency, target_poss, span_ids):
    temp = 0
    length = 0
    for span_idx in span_ids:
        for target_pos in range(*target_poss):
            temp += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()

        length += (span_idx[1] - span_idx[0])
    return temp/(target_poss[1] - target_poss[0]), length


def calculate_portions(saliency, evi_poss: List[Tuple[int, int]], attack_pos: List[Tuple[int, int]], emoji_pos: List[Tuple[int, int]], target_poss: Tuple[int, int], is_0k):
    """
    saliency: [batch_size, seq_len, seq_len] 倒数第二个位置对应prediction token

    target_poss: [l, r)
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
        
    saliency = saliency.numpy() #(seq_len, seq_len)
    np.fill_diagonal(saliency, 0)
    total_context_length = saliency.shape[1]

    topk_values, topk_indices = multi_torch_topk(saliency, target_poss, 200)

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []

    # proportion1: evidence -> target token
    proportion1 = 0
    evidence_length = 0
    for span_idx in evi_poss:
        temp_proportion1 = 0
        for target_pos in range(*target_poss):
            temp_proportion1 += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()
        proportion1 += temp_proportion1/(target_poss[1] - target_poss[0])
        
        # evidence proportions
        evidence_length += span_idx[1] - span_idx[0]
        temp_evidence_length = 0
        for target_pos in range(*target_poss):
            temp_evidence_length += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum() / (span_idx[1] - span_idx[0])
        
        evidence_proportions.append(temp_evidence_length/(target_poss[1] - target_poss[0]))

    # proportion2: all context -> target token

    temp_proportion2 = 0
    for target_pos in range(*target_poss):
        temp_proportion2 +=saliency[target_pos, :].sum()
    proportion2 = temp_proportion2/(target_poss[1] - target_poss[0])

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        temp_proportion3 = 0
        for target_pos in range(*target_poss):
            temp_proportion3 += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()

        proportion3 += temp_proportion3/(target_poss[1] - target_poss[0])
        
        irr_evidence_length += span_idx[1] - span_idx[0]


    # proportion4: remain context -> target token
    proportion4 = proportion2 - proportion1 - proportion3

    proportion5 = 0 #emoji context -> target token
    emoji_length = 0
    if emoji_pos:
        for span_idx in emoji_pos:
            temp_proportion5 = 0
            for target_pos in range(*target_poss):
                temp_proportion5 += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()
            proportion5 += temp_proportion5 / (target_poss[1] - target_poss[0])
            emoji_length += span_idx[1] -span_idx[0]

    else:
        emoji_length = 1

    if is_0k:
        proportion2 = (proportion1 + proportion3) /(evidence_length + irr_evidence_length)
    else:
        proportion2 = proportion2 / total_context_length

    proportion1 = proportion1 / evidence_length

    proportion3 = proportion3 / irr_evidence_length
    
    proportion5 = proportion5 / emoji_length
    if is_0k:
        proportion4 = 0.
    else:
        proportion4 = proportion4 / (total_context_length - evidence_length - irr_evidence_length)

    

    return proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_values, topk_indices

    
def calculate_embedding_portions(saliency, 
                                 evi_poss: List[Tuple[int, int]], 
                                 attack_pos: List[Tuple[int, int]], 
                                 emoji_pos: List[Tuple[int, int]],
                                 target_poss: Tuple[int, int],
                                 is_0k):
    """
    saliency: [batch_size, seq_len] 

    target_poss: [l, r)
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 1 or (len(saliency.shape) == 2 and saliency.shape[0] == 1)
    if len(saliency.shape) == 2:
        saliency = saliency.squeeze(0)
        
    saliency = saliency.numpy() #(seq_len, )

    total_context_length = saliency.shape[0]

    _, topk_indices = np_topk(saliency, 200)

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []
    
    # proportion1: evidence -> target token
    proportion1 = 0 # NOTE: proportion1: evidence token
    evidence_length = 0
    for span_idx in evi_poss:
        temp = saliency[np.array(range(*span_idx))].sum()
        proportion1 += temp
        # evidence proportions
        evidence_length += (span_idx[1] - span_idx[0])
        evidence_proportions += [temp/(span_idx[1] - span_idx[0])]

    # proportion2: all context -> target token
    proportion2 = saliency.sum()

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        proportion3 += saliency[np.array(range(*span_idx))].sum()
        irr_evidence_length += (span_idx[1] - span_idx[0])

    # proportion4: remain context -> target token
    proportion4 = proportion2 - proportion1 - proportion3

    proportion5 = 0 #emoji context -> target token
    emoji_length = 0
    if emoji_pos:
        for span_idx in emoji_pos:
            proportion5 += saliency[np.array(range(*span_idx))].sum()
            emoji_length += span_idx[1] -span_idx[0]
    else:
        emoji_length = 1


    if is_0k:
        proportion2 = (proportion1 + proportion3) / (evidence_length + irr_evidence_length)
    else:
        proportion2 = proportion2 / total_context_length

    proportion1 = proportion1 / evidence_length

    proportion3 = proportion3 / irr_evidence_length

    proportion5 = proportion5 / emoji_length

    if is_0k:
        proportion4 = 0.
    else:
        proportion4 = proportion4 / (total_context_length - evidence_length - irr_evidence_length)


    return proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices

def calculate_embedding_portions_update(saliency, 
                                 evi_poss: List[Tuple[int, int]], 
                                 attack_pos: List[Tuple[int, int]], 
                                 other_pos: List[Tuple[int, int]],
                                 emoji_pos: List[Tuple[int, int]],
                                 is_0k):
    """
    saliency: [batch_size, seq_len] 

    target_poss: [l, r)
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 1 or (len(saliency.shape) == 2 and saliency.shape[0] == 1)
    if len(saliency.shape) == 2:
        saliency = saliency.squeeze(0)
        
    saliency = saliency.numpy() #(seq_len, )

    total_context_length = saliency.shape[0]

    _, topk_indices = np_topk(saliency, 200)

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []
    
    # proportion1: evidence -> target token
    proportion1 = 0 # NOTE: proportion1: evidence token
    evidence_length = 0
    for span_idx in evi_poss:
        temp = saliency[np.array(range(*span_idx))].sum()
        proportion1 += temp
        # evidence proportions
        evidence_length += (span_idx[1] - span_idx[0])
        evidence_proportions += [temp/(span_idx[1] - span_idx[0])]

    # proportion2: all context -> target token
    proportion2 = saliency.sum()

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        proportion3 += saliency[np.array(range(*span_idx))].sum()
        irr_evidence_length += (span_idx[1] - span_idx[0])

    # proportion4: remain context -> target token
    # proportion4 = proportion2 - proportion1 - proportion3
    proportion4 = 0
    other_evidence_length = 0
    for span_idx in other_pos:
        proportion4 += saliency[np.array(range(*span_idx))].sum()
        other_evidence_length += (span_idx[1] - span_idx[0])

    # proportion5: emoji context -> target token
    proportion5 = 0 
    emoji_length = 0
    if emoji_pos:
        for span_idx in emoji_pos:
            proportion5 += saliency[np.array(range(*span_idx))].sum()
            emoji_length += (span_idx[1] - span_idx[0])
    else:
        emoji_length = 1


    if is_0k:
        proportion2 = (proportion1 + proportion3) / (evidence_length + irr_evidence_length)
    else:
        proportion2 = proportion2 / total_context_length

    proportion1 = proportion1 / evidence_length
    proportion3 = proportion3 / irr_evidence_length
    proportion4 = proportion4 / other_evidence_length
    proportion5 = proportion5 / emoji_length

    total = proportion1 + proportion3 + proportion4 + proportion5
    proportion1 = proportion1 / total
    proportion3 = proportion3 / total
    proportion4 = proportion4 / total
    proportion5 = 1 - proportion1 - proportion3 - proportion4

    # if is_0k:
    #     proportion4 = 0.
    # else:
    #     proportion4 = proportion4 / (total_context_length - evidence_length - irr_evidence_length)

    return proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices

def get_proportion_wla(saliency, class_poss, final_poss):
    saliency = saliency.detach().clone().cpu()
    class_poss = torch.hstack(class_poss).detach().clone().cpu()
    final_poss = final_poss.detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = saliency.numpy()
    np.fill_diagonal(saliency, 0)
    proportion1 = saliency[class_poss, :].sum()
    proportion2 = saliency[final_poss, class_poss].sum()
    proportion3 = saliency.sum() - proportion1 - proportion2

    N = int(final_poss)
    sum3 = (N + 1) * N / 2 - sum(class_poss) - len(class_poss)
    proportion1 = proportion1 / sum(class_poss)
    proportion2 = proportion2 / len(class_poss)
    proportion3 = proportion3 / sum3
    proportions = np.array([proportion1, proportion2, proportion3])
    return proportions


def find_multi_needle_idx(input_ids, tokenizer, needles, showlog = True):
    all_evi_pos = []
    for i, evi in enumerate(needles):
        if isinstance(evi, str):
            needle_ids = tokenizer(evi, add_special_tokens=False)["input_ids"]
        else:
            needle_ids = evi
        if showlog:
            logger.info(f"evidence {i} --> {tokenizer.decode(needle_ids, skip_special_tokens=False)}")
        span_len = len(needle_ids)
        for j in range(len(input_ids)):
            token_span = input_ids[j : j + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.99):
                all_evi_pos.append((j + 1, j + span_len))
                if showlog:
                    logger.info(f"find evidence {i} at --> {(j + 1, j + span_len)} --> {tokenizer.decode(input_ids[j + 1: j + span_len], skip_special_tokens=False)}")
                break
    return all_evi_pos


def test_model_with_adapter(model, input, golden, search_pos, attack_pos, emoji_pos, target_poss, is_0k, model_name, tokenizer, take_last_loss = True, with_adapter=False, start_layer = 0):

    embeddingmanager = ZeroEmbedManager(model, model_name, nonzero_poss=search_pos + attack_pos, factor = 0.1)
    attentionermanger = AttentionerManager(model, model_name, with_adapter=with_adapter, start_layer = start_layer)
    attentionermanger.zero_grad()

    output = model(input)
    if input.size(-1) == golden.size(-1):
        logits = output['logits'][:, :-1, :]
        labels = golden[:, 1:]
    else:
        logits = output['logits'][:, -1, :]
        labels = golden[:, -1]

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()

    ret = {}
    ret['loss'] = loss.detach().item()
    pros_dict = dict()
    pros_dict['grad'] = embeddingmanager.get_input_grad().tolist()

    for score_type in ["grad"]:
        saliencies = embeddingmanager.get_input_grad()

        proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices = calculate_embedding_portions(saliencies, search_pos, attack_pos, emoji_pos, target_poss, is_0k)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))

        pros_dict[score_type] = {
            'score': [proportion1, proportion3, proportion4, proportion5],
            "score_name":["Supporting","Interference","Irrelevant","Low-frequency"],
            "topk_indices": topk_indices,
            'topk_tokens': top_tokens,
            'evidence_proportions': evidence_proportions
        }

    ret['embedding'] = pros_dict


    pros_dict = dict()
    for i in trange(attentionermanger.start_layer, len(attentionermanger.attention_adapters)):
        pros_dict[i] = {}        
    
    for score_type in ["grad","weight","saliency"]:
        saliencies = getattr(attentionermanger, score_type)(use_abs=True)
        
        all_topk_values = []
        all_topk_indices= []
        for i in trange(attentionermanger.start_layer, len(attentionermanger.attention_adapters)):
            saliency = saliencies[i-attentionermanger.start_layer]        
            proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_values, topk_indices = calculate_portions(saliency, search_pos, attack_pos, emoji_pos, target_poss, is_0k)
            top_tokens = []
            for idx in topk_indices:
                top_tokens.append(tokenizer.decode(input[0][idx].item()))

            pros_dict[i][score_type] = {
                "score": [proportion1,  proportion3, proportion4, proportion5],
                "score_name":["Supporting","Interference","Irrelevant","Low-frequency"], "topk_values":topk_values,
                                        'topk_indices':topk_indices, 'topk_tokens': top_tokens, 'evidence_proportions': evidence_proportions}
        
            all_topk_values.append(topk_values)
            all_topk_indices.append(topk_indices)
        
        max_indice = max([k.max() for k in all_topk_indices])
        lines = np.zeros(max_indice + 1)
        for topk_v, topk_i, in zip(all_topk_values,all_topk_indices):
            lines [topk_i] += topk_v
        _, all_layer_topk_indices = np_topk(lines, 200)
        
        all_layer_topk_tokens = [tokenizer.decode(input[0][idx].item()) for idx in all_layer_topk_indices]

        pros_dict[score_type]= {"topk_tokens": all_layer_topk_tokens,
                                "topk_indices": all_layer_topk_indices}

    ret['attention'] = pros_dict
    ret['evidence_pos'] = search_pos
    ret['attack_pos'] = attack_pos
    ret['emoji_pos'] = emoji_pos
    ret['irr_length'] = embeddingmanager.get_input_grad().flatten().size(0) - sum([x[1]-x[0] for x in search_pos + attack_pos + emoji_pos])

    del embeddingmanager
    del attentionermanger
    return ret

def test_model_with_adapter_update(model, input, search_pos, attack_pos, other_pos, emoji_pos, is_0k, model_name, tokenizer, take_last_loss = True, with_adapter=False, start_layer = 0):

    embeddingmanager = ZeroEmbedManager(model, model_name, nonzero_poss=search_pos + attack_pos, factor = 0.1)
    attentionermanger = AttentionerManager(model, model_name, with_adapter=with_adapter, start_layer = start_layer)
    attentionermanger.zero_grad()

    output = model(input)
    loss = (- output['logits'][0, -1, :].max())
    loss.backward()

    ret = {}
    ret['loss'] = loss.detach().item()
    pros_dict = dict()
    pros_dict['grad'] = embeddingmanager.get_input_grad().tolist()

    for score_type in ["grad"]:
        saliencies = embeddingmanager.get_input_grad()

        proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices = calculate_embedding_portions_update(saliencies, search_pos, attack_pos, other_pos, emoji_pos, is_0k)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))

        pros_dict[score_type] = {
            'score': [proportion1, proportion3, proportion4, proportion5],
            "score_name":["Supporting","Interference","Irrelevant","Low-frequency"],
            "topk_indices": topk_indices,
            'topk_tokens': top_tokens,
            'evidence_proportions': evidence_proportions
        }

    ret['embedding'] = pros_dict


    pros_dict = dict()
    for i in trange(attentionermanger.start_layer, len(attentionermanger.attention_adapters)):
        pros_dict[i] = {}        
    
    for score_type in ["grad","weight","saliency"]:
        saliencies = getattr(attentionermanger, score_type)(use_abs=True)
        
        all_topk_values = []
        all_topk_indices= []
        for i in trange(attentionermanger.start_layer, len(attentionermanger.attention_adapters)):
            saliency = saliencies[i-attentionermanger.start_layer]        
            proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_values, topk_indices = calculate_portions(saliency, search_pos, attack_pos, emoji_pos, target_poss, is_0k)
            top_tokens = []
            for idx in topk_indices:
                top_tokens.append(tokenizer.decode(input[0][idx].item()))

            pros_dict[i][score_type] = {
                "score": [proportion1,  proportion3, proportion4, proportion5],
                "score_name":["Supporting","Interference","Irrelevant","Low-frequency"], "topk_values":topk_values,
                                        'topk_indices':topk_indices, 'topk_tokens': top_tokens, 'evidence_proportions': evidence_proportions}
        
            all_topk_values.append(topk_values)
            all_topk_indices.append(topk_indices)
        
        max_indice = max([k.max() for k in all_topk_indices])
        lines = np.zeros(max_indice + 1)
        for topk_v, topk_i, in zip(all_topk_values,all_topk_indices):
            lines [topk_i] += topk_v
        _, all_layer_topk_indices = np_topk(lines, 200)
        
        all_layer_topk_tokens = [tokenizer.decode(input[0][idx].item()) for idx in all_layer_topk_indices]

        pros_dict[score_type]= {"topk_tokens": all_layer_topk_tokens,
                                "topk_indices": all_layer_topk_indices}

    ret['attention'] = pros_dict
    ret['evidence_pos'] = search_pos
    ret['attack_pos'] = attack_pos
    ret['emoji_pos'] = emoji_pos
    ret['irr_length'] = embeddingmanager.get_input_grad().flatten().size(0) - sum([x[1]-x[0] for x in search_pos + attack_pos + emoji_pos])

    del embeddingmanager
    del attentionermanger
    return ret

def test_model_ig(model, input, golden, search_pos, attack_pos, emoji_pos, target_poss, is_0k, model_name, tokenizer):

    embeddingmanager = ZeroEmbedManager(model, model_name, nonzero_poss=search_pos + attack_pos, factor = 0.1)

    output = model(input)
    if input.size(-1) == golden.size(-1):
        logits = output['logits'][:, :-1, :]
        labels = golden[:, 1:]
    else:
        logits = output['logits'][:, -1, :]
        labels = golden[:, -1]

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()

    ret = {}
    ret['loss'] = loss.detach().item()
    pros_dict = dict()
    pros_dict['grad'] = embeddingmanager.get_input_grad().tolist()

    for score_type in ["grad"]:
        saliencies = embeddingmanager.get_input_grad()

        proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices = calculate_embedding_portions(saliencies, search_pos, attack_pos, emoji_pos, target_poss, is_0k)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))

        pros_dict[score_type] = {
            'score': [proportion1, proportion3, proportion4, proportion5],
            "score_name":["Supporting","Interference","Irrelevant","Low-frequency"],
            "topk_indices": topk_indices,
            'topk_tokens': top_tokens,
            'evidence_proportions': evidence_proportions
        }

    ret['embedding'] = pros_dict

    ret['evidence_pos'] = search_pos
    ret['attack_pos'] = attack_pos
    ret['emoji_pos'] = emoji_pos
    ret['irr_length'] = embeddingmanager.get_input_grad().flatten().size(0) - sum([x[1]-x[0] for x in search_pos + attack_pos + emoji_pos])

    del embeddingmanager
    return ret

def test_model_ig_update(model, input, search_pos, attack_pos, other_pos, emoji_pos, is_0k, model_name, tokenizer):

    embeddingmanager = ZeroEmbedManager(model, model_name, nonzero_poss=search_pos + attack_pos, factor = 0.1)

    # output = model(input)
    # if input.size(-1) == golden.size(-1):
    #     logits = output['logits'][:, :-1, :]
    #     labels = golden[:, 1:]
    # else:
    #     logits = output['logits'][:, -1, :]
    #     labels = golden[:, -1]
    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    output = model(input)
    logits_first = output['logits'][:, -1, :] / 4
    y_pred = torch.argmax(logits_first, dim=-1)
    loss = F.cross_entropy(logits_first, y_pred)
    print(f"Logits is:{logits_first}")
    print(f"Y_pred is:{y_pred}")
    print(f"Loss is:{loss}")
    loss.backward()

    probs = torch.softmax(logits_first, dim=-1)
    next_token_id = torch.argmax(probs).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"Next token is:{next_token}")

    ret = {}
    ret['loss'] = loss.detach().item()
    pros_dict = dict()
    grad_dict = embeddingmanager.get_input_grad()

    for i in range(len(grad_dict)):
        saliencies = grad_dict[i]
        pros_dict[i] = {}

        proportion1, proportion2, proportion3, proportion4, proportion5, evidence_proportions, topk_indices = calculate_embedding_portions_update(saliencies, search_pos, attack_pos, other_pos, emoji_pos, is_0k)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))

        pros_dict[i]["grad"] = {
            'score': [proportion1, proportion3, proportion4, proportion5],
            "score_name":["Supporting","Interference","Irrelevant","Low-frequency"],
            "topk_indices": topk_indices,
            'topk_tokens': top_tokens,
            'evidence_proportions': evidence_proportions
        }

    ret['embedding'] = pros_dict

    ret['evidence_pos'] = search_pos
    ret['attack_pos'] = attack_pos
    ret['other_pos'] = other_pos
    ret['emoji_pos'] = emoji_pos
    # ret['irr_length'] = embeddingmanager.get_input_grad().flatten().size(0) - sum([x[1]-x[0] for x in search_pos + attack_pos + emoji_pos])

    del embeddingmanager
    return ret

def random_combine(ref:list, att:list, return_snd_pos = False, seed = None):
    if seed is not None:
        random.seed(seed)
    att_list =[[] for _ in range(len(ref) + 1)]
    for p_att in att[:-1]:
        att_list[random.randint(0,len(ref)-1)].append(p_att)
    att_list[-1].append(att[-1])

    results = [k for k in att_list[0]]

    if return_snd_pos:
        insert_pos = list(range(len(results)))
    for r, patt in zip(ref,att_list[1:]):
        results.append(r)
        if return_snd_pos:
            insert_pos.extend(list(range(len(results), len(results) + len(patt))))

        results.extend(patt)
            
    if return_snd_pos:
        assert len(att) == len(insert_pos)
        return results, insert_pos

    return results



def get_random_emoji(tokenizer, num=50, return_idx=True, seed=None):
    all_emojis = list(emoji.EMOJI_DATA.keys())  # get all emojis
    if seed is not None:
        random.seed(seed)
    random_emojis = random.sample(all_emojis, num)
    print(f"your chose emoji: {random_emojis}")
    if return_idx:
        index_emojis = []
        for e in random_emojis:
            index_emojis.append(tokenizer(e, add_special_tokens=False).input_ids)
        return index_emojis
    return random_emojis

def analyze_context_groups(context_tokens: List[str]) -> Dict[str, List[int]]:
    """
    分析context tokens并分组
    
    Args:
        context_tokens: context中的token文本列表
        
    Returns:
        包含各组token索引的字典
    """
    # 将token列表转换为文本进行分析
    full_text = ''.join(context_tokens)
    
    # 定义分组边界的关键词
    groups = {
        'System Instruction': [],
        'Raw Experience': [],
        'Condensed Experience': [],
        'Current Trajectory': [],
        'Others': []
    }
    
    # 定义分组边界的关键词（简化版本，不需要END OF EXAMPLES）
    boundary_keywords = [
        'HereĠareĠtwoĠexamples:',
        'TheĠfollowingĠareĠsomeĠexperience',
        'NowĠit\'sĠyourĠturn!'
    ]
    
    # 在文本中查找边界位置
    boundary_positions = []
    for keyword in boundary_keywords:
        pos = full_text.find(keyword)
        # 如果找不到，尝试查找带特殊字符的版本
        if pos == -1:
            # 尝试不同的特殊字符组合
            variations = [
                f'Ġ{keyword}',  # 空格前缀
                f'▁{keyword}',  # 下划线前缀
                keyword.replace(' ', 'Ġ'),  # 空格替换为Ġ
                keyword.replace(' ', '▁'),  # 空格替换为▁
            ]
            for variation in variations:
                pos = full_text.find(variation)
                if pos != -1:
                    break
        boundary_positions.append(pos)
    
    # 将文本位置转换为token位置
    token_boundaries = []
    current_char_pos = 0
    
    for i, token in enumerate(context_tokens):
        token_start = current_char_pos
        token_end = current_char_pos + len(token)
        
        # 检查每个边界关键词是否在当前token范围内
        for j, text_pos in enumerate(boundary_positions):
            if text_pos != -1 and token_start <= text_pos < token_end:
                token_boundaries.append((j, i))  # (boundary_index, token_index)
        
        current_char_pos = token_end
    
    # 设置边界位置
    system_end_token = -1      # HereĠareĠtwoĠexamples:
    condensed_start_token = -1 # TheĠfollowingĠareĠsomeĠexperience
    current_traj_start_token = -1 # NowĠit\'sĠyourĠturn!
    
    for boundary_idx, token_idx in token_boundaries:
        if boundary_idx == 0:  # 'HereĠareĠtwoĠexamples:'
            system_end_token = token_idx
        elif boundary_idx == 1:  # 'TheĠfollowingĠareĠsomeĠexperience'
            condensed_start_token = token_idx
        elif boundary_idx == 2:  # 'NowĠit\'sĠyourĠturn!'
            current_traj_start_token = token_idx
    
    # 如果找不到边界，使用默认分组（基于文本长度比例）
    if system_end_token == -1:
        system_end_token = len(context_tokens) // 4
    if condensed_start_token == -1:
        condensed_start_token = len(context_tokens) * 3 // 5
    if current_traj_start_token == -1:
        current_traj_start_token = len(context_tokens)
    
    # 分配token到各组
    for i in range(len(context_tokens)):
        if i < system_end_token:
            groups['System Instruction'].append(i)
        elif i < condensed_start_token:
            groups['Raw Experience'].append(i)
        elif i < current_traj_start_token:
            groups['Condensed Experience'].append(i)
        else:
            groups['Current Trajectory'].append(i)
    
    return groups