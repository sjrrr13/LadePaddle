import os
import random

import paddle
from paddlenlp.transformers.generation_utils import LogitsProcessorList
import paddle.nn.functional as F


FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))


def greedy_search_proxy(self, *args, **kwargs):
    USE_LADE = int(os.environ.get("USE_LADE", 0))
    CHAT = int(os.environ.get("CHAT", 0))
    if CHAT and USE_LADE:
        # return jacobi_greedy_search_multilevel(self, chat=True, *args, **kwargs)
    # elif CHAT:
        print("CHAT is not supported yet.")
    
    if USE_LADE:
        return jacobi_greedy_search_multilevel(self, *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)


def sample_proxy(self, *args, **kwargs):
    USE_LADE = int(os.environ.get("USE_LADE", 0))
    
    if USE_LADE:
        return jacobi_sample_multilevel(self, *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)


# least recently used
# most recently added
# least recently added
# most used
def update_lra(token_map, key_token, value_tup, GUESS_SET_SIZE):
    # The least recently added N-gram will be removed if the pool is full.
    if key_token not in token_map:
        token_map[key_token] = []
    if value_tup in token_map[key_token]:
        token_map[key_token].remove(value_tup)
        token_map[key_token].append(value_tup)
    elif len(token_map[key_token]) < GUESS_SET_SIZE:
        token_map[key_token].append(value_tup)
    else:
        assert len(token_map[key_token]) == GUESS_SET_SIZE
        token_map[key_token] = token_map[key_token][1:] + [value_tup]


def update_fifo(token_map, key_token, value_tup, GUESS_SET_SIZE):
    if key_token not in token_map:
        token_map[key_token] = []
    if value_tup in token_map[key_token]:
        return
    elif len(token_map[key_token]) < GUESS_SET_SIZE:
        token_map[key_token].append(value_tup) 
    else:
        assert len(token_map[key_token]) == GUESS_SET_SIZE
        token_map[key_token] = token_map[key_token][1:] + [value_tup]


def update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE):
    # Update N-gram pool with newly generated N-grams in 2D window
    if GUESS_SET_SIZE != -1:
        # if lst_token not in token_map:
        #     token_map[lst_token] = []
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        # if tup in token_map[lst_token]:
        #     token_map[lst_token].remove(tup)
        #     token_map[lst_token].append(tup)
        # elif len(token_map[lst_token]) < GUESS_SET_SIZE:
        #     token_map[lst_token].append(tup) 
        # else:
        #     assert len(token_map[lst_token]) == GUESS_SET_SIZE
        #     token_map[lst_token] = token_map[lst_token][1:] + [tup]
        # update_lra(token_map, lst_token, tup, GUESS_SET_SIZE)
        update_fifo(token_map, lst_token, tup, GUESS_SET_SIZE)

        for i in range(1, WINDOW_SIZE):
        #     if past_tokens[0][i - 1] not in token_map:
        #         token_map[past_tokens[0][i - 1]] = []
            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)
        #     if tup in token_map[past_tokens[0][i - 1]]:
        #         token_map[past_tokens[0][i - 1]].remove(tup)
        #         token_map[past_tokens[0][i - 1]].append(tup)
        #     elif len(token_map[past_tokens[0][i - 1]]) < GUESS_SET_SIZE:
        #         token_map[past_tokens[0][i - 1]].append(tup) 
        #     else:
        #         assert len(token_map[past_tokens[0][i - 1]]) == GUESS_SET_SIZE
        #         token_map[past_tokens[0][i - 1]] = token_map[past_tokens[0][i - 1]][1:] + [tup]
            # update_lra(token_map, past_tokens[0][i - 1], tup, GUESS_SET_SIZE)
            update_fifo(token_map, past_tokens[0][i - 1], tup, GUESS_SET_SIZE)
    else:
        if lst_token not in token_map:
            token_map[lst_token] = set()
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        token_map[lst_token].add(tup) 

        for i in range(1, WINDOW_SIZE):
            if past_tokens[0][i - 1] not in token_map:
                token_map[past_tokens[0][i - 1]] = set()
            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)
            token_map[past_tokens[0][i - 1]].add(tup) 


def append_new_generated_to_pool(tokens, token_map, LEVEL, GUESS_SET_SIZE):
    # If a N-gram is used, update it as the last N-gram in the pool. This function implements LRU policy.
    if len(tokens) != LEVEL:
        return 
    lst_token = tokens[0]
    tup = tuple(tokens[1:])

    if GUESS_SET_SIZE != -1:
        # if lst_token not in token_map:
        #     token_map[lst_token] = []
        # if tup in token_map[lst_token]:
        #     token_map[lst_token].remove(tup)
        #     token_map[lst_token].append(tup)
        # elif len(token_map[lst_token]) < GUESS_SET_SIZE:
        #     token_map[lst_token].append(tup) 
        # else:
        #     assert len(token_map[lst_token]) == GUESS_SET_SIZE
        #     token_map[lst_token] = token_map[lst_token][1:] + [tup]
        # update_lra(token_map, lst_token, tup, GUESS_SET_SIZE)
        update_fifo(token_map, lst_token, tup, GUESS_SET_SIZE)
    else:
        if lst_token not in token_map:
            token_map[lst_token] = set()
        token_map[lst_token].add(tup) 


def fill_pool_with_prompt(prompts, token_map, LEVEL, GUESS_SET_SIZE):
    for start_idx in range(len(prompts) - LEVEL + 1):
        lst_token = prompts[start_idx]
        tup = tuple(prompts[start_idx + 1: start_idx + LEVEL])
        
        if len(tup) != LEVEL - 1:
            return 
        
        if GUESS_SET_SIZE != -1:
            # if lst_token not in token_map:
            #     token_map[lst_token] = []
            
            # if tup in token_map[lst_token]:
            #     token_map[lst_token].remove(tup)
            #     token_map[lst_token].append(tup)
            # elif len(token_map[lst_token]) < GUESS_SET_SIZE:
            #     token_map[lst_token].append(tup) 
            # else:
            #     assert len(token_map[lst_token]) == GUESS_SET_SIZE
            #     token_map[lst_token] = token_map[lst_token][1:] + [tup]
            # update_lra(token_map, lst_token, tup, GUESS_SET_SIZE)
            update_fifo(token_map, lst_token, tup, GUESS_SET_SIZE)
        else:
            if lst_token not in token_map:
                token_map[lst_token] = set()
            token_map[lst_token].add(tup) 


def filter_window(level_window, eos_token_id, reset_func):
    """
    Reset the tokens in level_window that are equal to eos_token_id.
    Baecuse too many <EOS> in window will lead to numerical error.
    """
    for idx in range(len(level_window)):
        if level_window[idx] == eos_token_id:
            level_window[idx] = reset_func()


def TopKProcess(probs, top_k, min_tokens_to_keep):
    top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
    # Remove all tokens with a probability less than the last token of the top-k
    # cast to float16 to support generation & d2s
    if probs.dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)
        topk_probs, _ = paddle.topk(probs, k=top_k)
        topk_probs = paddle.cast(topk_probs, paddle.bfloat16)
    else:
        topk_probs, _ = paddle.topk(probs, k=top_k)

    probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, -float('Inf')))
    return probs


def TopPProcess(probs, top_p, min_tokens_to_keep):
    org_dtype = probs.dtype
    if org_dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)
    sorted_indices = paddle.argsort(probs, descending=True)

    if isinstance(sorted_indices, tuple):
        sorted_probs, sorted_indices = sorted_indices
    else:
        sorted_probs = paddle.sort(probs, descending=True)

    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Set 'min_tokens_to_keep - 1' because the first token is kept
        sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
    )

    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, -float('Inf')), probs)
    if org_dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.bfloat16)
    return probs


def process_logits(logits, temperature, top_k, top_p, min_tokens_to_keep):
    if temperature is not None and temperature > 0.0 and temperature <= 1.0:
        probs = logits / temperature
    if top_k is not None and top_k >= 1:
        probs = TopKProcess(probs, top_k, min_tokens_to_keep)
    if top_p is not None and top_p >= 0 and top_p <= 1.0:
        probs = TopPProcess(probs, top_p, min_tokens_to_keep)
    return probs


def jacobi_sample_multilevel(
    self,
    input_ids,
    logits_processor,
    max_length,
    pad_token_id,
    eos_token_id,
    top_k=None,
    top_p=None,
    temperature=None,
    min_tokens_to_keep=1,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    return_dict_in_generate=False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
    ```"""

    model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = paddle.to_tensor(eos_token_id, place=input_ids.place) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else False
    output_attentions = (
        output_attentions if output_attentions is not None else False
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else False
    )

    # ** LADE do not support `return_dict_in_generate`
    return_dict_in_generate = False
    chat = False

    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 7)
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 7)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    LEVEL = CONFIG_MAP.get("LEVEL", 5)
    DEBUG = CONFIG_MAP.get("DEBUG", 0)
    LOCAL_RANK = CONFIG_MAP.get("LOCAL_RANK", 0)
    USE_FLASH = CONFIG_MAP.get("USE_FLASH", 0) # Not use flash by default
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 0)
    GUESS_SIZE = LEVEL - 1  # N-1
    NOT_SEQ = 0
    CONTINUE_ALL = 0
    
    all_old_tokens = input_ids[0].tolist()
    _init_len = len(all_old_tokens)
    init_len = _init_len
    
    def set_token():
        return random.choice(all_old_tokens)

    # `past_token`s is a 2D window for efficient N-gram generation with shape [W, N-1]
    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]
    
    fill_level = 0      # How many windows in `past_tokens` has been filled
    guess_tokens = None # List of N-grams start with `lst_token`
    token_map = {}      # N-gram pool
    steps = 0           # Decoding setps
    guess_skip_dist = 0 # Not been used in sampling

    if POOL_FROM_PROMPT:
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)

    if chat:
        pass

    # Auto-regressive generation
    while init_len < max_length:
        # **1. Prepare inputs
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        # **2. Find N-grams start with `lst_token`, sequeeze into 1D list
        if past_tokens[LEVEL - 2] is not None and lst_token in token_map and GUESS_SET_SIZE > 0:  
            guess_tokens_ = token_map[lst_token]
            guess_tokens = []
            for tok in list(guess_tokens_):
                guess_tokens += list(tok)
        else:
            guess_tokens = None

        # **3. Forward
        outputs = self.jforward_multilevel(
            **model_inputs,
            past_tokens=past_tokens,
            guess_tokens=guess_tokens,
            return_dict=True,
            not_seq=NOT_SEQ,
            continue_all=CONTINUE_ALL,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            level=LEVEL,
            WINDOWS_SIZE=WINDOW_SIZE,
            guess_size=GUESS_SIZE,
            fill_level=fill_level,
            la_mask_offset=0,
            use_flash=USE_FLASH
        )
        steps += 1

        # **4. Process logits, the same as `sampel` in generation_utils.py in PaddleNLP
        logits = outputs.out_logits # In PaddleNLP, logits = outputs.logits[:, -1, :]
        probs = process_logits(logits, temperature, top_k, top_p, min_tokens_to_keep)

        # **5. Verification
        max_hit = 0 # Num of extra tokens accepted in one step
        if fill_level < LEVEL - 2:  # Fill 2D window, not use verification branch
            probs = F.softmax(probs, axis=-1)
            next_tokens = paddle.multinomial(probs, num_samples=1).squeeze(1)
            hits = [next_tokens.item()] 
            
            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:] 
            past_tokens[fill_level + 1] = paddle.argmax(outputs.inp_logits, axis=-1)[0].tolist()
            if fill_level > 0:
                past_tokens[fill_level + 1] = past_tokens[fill_level + 1][1:]

            fill_level += 1
        else:
            if guess_tokens is not None:    # Verify N-grams start with lst_token
                hits = []
                probs_next = F.softmax(probs, axis=-1)   # Probs of next token verified
                probs_next = probs_next[0]
                # Logits of N-grams, shape of guess_logits is [num_Ngrams, vocab_size]
                guess_logits = process_logits(outputs.guess_logits[0], temperature, top_k, top_p, min_tokens_to_keep)
                guess_probs = F.softmax(guess_logits, axis=-1)
                guess_indices = list(range(outputs.guess_logits.shape[1] // GUESS_SIZE)) # range(num_Ngrams)

                for idx_in_ngram in range(GUESS_SIZE):  # Progressively verify along the N-gram length
                    g_idx = 0
                    is_accept = False
                    
                    while g_idx < len(guess_indices):
                        guess_idx = guess_indices[g_idx]
                        draft_token = guess_tokens[guess_idx * GUESS_SIZE + idx_in_ngram] # Lookahead drafts, guess_tokens is (N-1) * k
                        draft_prob = min(1, probs_next[draft_token].item())
                        gamma = random.random()

                        if gamma < draft_prob:   # Accept, update all potential speculations and probabilities
                            hits.append(draft_token)
                            is_accept = True 
                            max_hit_idx = guess_idx
                            new_guess_indices = []
                            for idx in guess_indices:   # Remove N-grams with mismatched prefixes
                                idx_draft_token = guess_tokens[idx * GUESS_SIZE + idx_in_ngram]
                                if idx_draft_token == draft_token:
                                    new_guess_indices.append(idx)
                            guess_indices = new_guess_indices
                            break 
                        else:   # Reject, go to next speculation
                            probs_next[draft_token] = 0
                            probs_next = probs_next / probs_next.sum()
                            g_idx += 1
                    
                    if is_accept:
                        probs_next = guess_probs[guess_idx * GUESS_SIZE + idx_in_ngram] # Update probs of next token
                        continue 
                    else:   # Guarantee one step movement
                        new_token_gen = paddle.multinomial(probs_next, num_samples=1).item()    # ValueError: (InvalidArgument) Each element of multinomial'input must >= 0, but got nan.
                        hits.append(new_token_gen)
                        break
                max_hit = len(hits) - 1
            else:   # No matched N-grams
                probs_next = F.softmax(probs, axis=-1)
                next_tokens = paddle.multinomial(probs_next, num_samples=1).squeeze(1)
                hits = [next_tokens.item()]

            new_results = paddle.argmax(outputs.inp_logits, axis=-1)[0].tolist()    # Generated N-grams in next step with greedy search
            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE
            update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)

            # Update 2D window
            if ALWAYS_FWD_ONE:  # Only 1 step movement
                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:   # max_hit steps movement
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]

            if max_hit > 0:
                if not ALWAYS_FWD_ONE:
                    for level in range(LEVEL - 1):
                        past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([1, max_hit], dtype=attention_mask.dtype)]   # place=attention_mask.place
                    , axis=1)

            if eos_token_id is not None:    # Filter <EOS> 
                filter_window(past_tokens[LEVEL - 2], eos_token_id[0], set_token)

        past_key_values = []    # Update kv cache of correctly speculated tokens
        for idx, kv in enumerate(outputs.past_key_values):
            for h in range(max_hit):
                assert outputs.step_len == kv[0].shape[1], f"{outputs.step_len} != {kv[0].shape[1]}, {kv[0].shape}"
                cache_len = outputs.step_len - len(guess_tokens) + max_hit_idx * GUESS_SIZE + h
                kv[0][:, outputs.kvcache_len + h, :, :] = kv[0][:, cache_len, :, :]
                kv[1][:, outputs.kvcache_len + h, :, :] = kv[1][:, cache_len, :, :]
            past_key_values.append((
                kv[0][:, :outputs.kvcache_len + max_hit, :, :], 
                kv[1][:, :outputs.kvcache_len + max_hit, :, :]))
        outputs.past_key_values = past_key_values

        lst_token = hits[max_hit]

        for hit_ids in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_ids] == eos_token_id[0]:   # If reach <EOS>
                all_old_tokens.append(hits[hit_ids])
                next_tokens = eos_token_id_tensor
                max_hit = hit_ids
                break
            else:
                all_old_tokens.append(hits[hit_ids])
                if POOL_FROM_PROMPT:
                    append_new_generated_to_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)

        init_len = len(all_old_tokens)
        
        if chat:
            pass
        
        # Update generated ids, model inputs, and length for next step
        input_ids = paddle.concat(
            [input_ids, paddle.to_tensor(hits[:max_hit + 1], place=input_ids.place, dtype=input_ids.dtype).unsqueeze(0)]
            , axis=-1)
        model_kwargs = self.update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

    compression = round((len(all_old_tokens) - _init_len) / steps, 2)
    if DEBUG and LOCAL_RANK == 0:
        CONFIG_MAP["log"].append([len(all_old_tokens) - init_len, steps, compression])
    
    if return_dict_in_generate:
        pass
    else:
        return input_ids, compression


def jacobi_greedy_search_multilevel(
    self,
    input_ids,
    logits_processor,
    max_length=None,
    pad_token_id=None,
    eos_token_id=None,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    return_dict_in_generate=False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    
    model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = paddle.to_tensor(eos_token_id, place=input_ids.place) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else False
    output_attentions = (
        output_attentions if output_attentions is not None else False
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else False
    )
    return_dict_in_generate = False
    chat = False

    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 7)
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 7)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    LEVEL = CONFIG_MAP.get("LEVEL", 5)
    DEBUG = CONFIG_MAP.get("DEBUG", 0)
    LOCAL_RANK = CONFIG_MAP.get("LOCAL_RANK", 0)
    USE_FLASH = CONFIG_MAP.get("USE_FLASH", 0) # Not use flash by default
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 0)
    USE_AWQ = False
    GUESS_SIZE = LEVEL - 1
    NOT_SEQ = 0
    CONTINUE_ALL = 0
    TEMP_FOR_GUESS = 0.0
    USE_AWQ = False 
    windows = [WINDOW_SIZE]

    assert TEMP_FOR_GUESS == 0
    assert ALWAYS_FWD_ONE == 1
    assert USE_AWQ == False 

    all_old_tokens = input_ids[0].tolist()
    _init_len = len(all_old_tokens)
    seq_len = _init_len
    order_copy_from_idx = [0]

    def random_set():
        return random.randint(0, self.vocab_size - 1)

    def copy_from():
        return random.choice(all_old_tokens)

    def order_copy_from():
        if order_copy_from_idx[0] >= len(all_old_tokens):
            order_copy_from_idx[0] = 0
        ret = all_old_tokens[order_copy_from_idx[0]]
        order_copy_from_idx[0] = 1 + order_copy_from_idx[0]
        return ret

    def copy_from_last():
        return all_old_tokens[-1]

    set_token = copy_from

    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]

    fill_level = 0
    guess_tokens = None
    token_map = {}
    steps = 0
    guess_skip_dist = 0

    if POOL_FROM_PROMPT:
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)
    
    if chat:
        pass

    count1, count2 = 1, 1
    while seq_len < max_length:
        # **1. Prepare inputs
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        # **2. Find N-grams start with `lst_token`, sequeeze into 1D list
        if past_tokens[LEVEL - 2] is not None and lst_token in token_map:
            if GUESS_SET_SIZE == -1:
                _guess_tokens = list(token_map[lst_token])  
            else:
                _guess_tokens = token_map[lst_token]
            guess_tokens = []
            for tok in list(_guess_tokens):
                guess_tokens += list(tok)
        else:
            guess_tokens = None

        assert return_dict_in_generate == False

        past_tokens_inp = past_tokens
            
        outputs = self.jforward_multilevel(
            **model_inputs,
            past_tokens=past_tokens_inp,
            guess_tokens=guess_tokens,
            return_dict=True,
            not_seq=NOT_SEQ,
            continue_all=CONTINUE_ALL,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            level=LEVEL,
            WINDOWS_SIZE=WINDOW_SIZE,
            guess_size=GUESS_SIZE,
            fill_level=fill_level,
            la_mask_offset=0,
            use_flash=USE_FLASH
        )
        steps += 1

        next_token_logits = outputs.out_logits

        # **4. Process logits, the same as `sampel` in generation_utils.py in PaddleNLP
        next_tokens_scores = next_token_logits
        next_tokens = paddle.argmax(next_tokens_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        
        first_guess = next_tokens.item()
        max_hit = 0 #  Length of longest accepted sequence in verification branch 
        hits = [first_guess] + [0] * (GUESS_SIZE - 1)
        new_results = []

        if fill_level < LEVEL - 2:
            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:]
            past_tokens[fill_level + 1] = paddle.argmax(outputs.inp_logits, axis=-1)[0].tolist()
            if fill_level > 0:
                past_tokens[fill_level + 1] = past_tokens[fill_level + 1][1:]

            fill_level += 1
        else: 
            if guess_tokens is not None:
                guess_results = paddle.argmax(outputs.guess_logits, axis=-1)[0].tolist()
                for guess_idx in range(len(guess_results) // GUESS_SIZE):
                    guess_offset = guess_idx * GUESS_SIZE
                    correct = [first_guess] + guess_results[guess_offset : guess_offset + GUESS_SIZE]
                    myguess = guess_tokens[guess_offset : guess_offset + GUESS_SIZE]
                    idx = 0 # N-gram index
                    for idx in range(len(myguess)):
                        if myguess[idx] != correct[idx]:
                            break 
                    if idx > max_hit:
                        max_hit = idx 
                        max_hit_idx = guess_idx 
                        hits[:max_hit + 1] = correct[:max_hit + 1]
            
            new_results = paddle.argmax(outputs.inp_logits, axis=-1)[0].tolist()
            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE
            update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)

            assert ALWAYS_FWD_ONE
            if ALWAYS_FWD_ONE:
                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]

        if max_hit > 0:
            if not ALWAYS_FWD_ONE:
                for level in range(LEVEL - 1):
                    past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [attention_mask, paddle.ones([1, max_hit], dtype=attention_mask.dtype)]
                , axis=1)

        # Update kv-cache from verification branch  
        past_key_values = []
        cache_offset = outputs.step_len - len(guess_tokens) + max_hit_idx * GUESS_SIZE if max_hit > 0 else 0
        cache_len = outputs.kvcache_len
        for idx, kv in enumerate(outputs.past_key_values):
            if max_hit > 0:
                kv[0][:, cache_len:cache_len + max_hit, :, :] = kv[0][:, cache_offset:cache_offset + max_hit, :, :]
                kv[1][:, cache_len:cache_len + max_hit, :, :] = kv[1][:, cache_offset:cache_offset + max_hit, :, :]
            past_key_values.append((
                kv[0][:, :cache_len + max_hit, :, :], 
                kv[1][:, :cache_len + max_hit, :, :]))
        outputs.past_key_values = past_key_values

        lst_token = hits[max_hit]

        for hit_idx in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_idx] == eos_token_id[0]:
                all_old_tokens.append(hits[hit_idx])
                next_tokens = eos_token_id_tensor
                max_hit = hit_idx
                break
            else:
                all_old_tokens.append(hits[max_hit])
                # # LRU policy
                # if POOL_FROM_PROMPT:
                #     append_new_generated_to_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)

        seq_len = len(all_old_tokens)
        windows.append(WINDOW_SIZE)
        
        # Update WINDOW_SIZE and past_tokens according to seq length
        if (seq_len-_init_len) >= 100:
            if (seq_len-_init_len) >= 50*count1 and WINDOW_SIZE > 2:
                WINDOW_SIZE -= 1
                count1 += 1
                for i in range(GUESS_SIZE):
                    past_tokens[i] = past_tokens[i][:-1]
        else:
            if (seq_len-_init_len) >= 25*count2 and fill_level >= LEVEL - 2:
                WINDOW_SIZE += 1
                count2 += 1
                # Choose a new token from N-gram pool
                new_window_token = random.choice(list(token_map.keys()))
                past_tokens[0].append(new_window_token)
                for i in range(fill_level):
                    past_tokens[i+1].append(list(token_map[new_window_token])[0][i])

        if chat and LOCAL_RANK == 0:
            pass
        
        input_ids = paddle.concat(
            [input_ids, paddle.to_tensor(hits[:max_hit + 1], dtype=input_ids.dtype).unsqueeze(0)]
            , axis=-1)
        model_kwargs = self.update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

    compression = round((len(all_old_tokens) - _init_len) / steps, 2)
    if DEBUG and LOCAL_RANK == 0:
        CONFIG_MAP["log"].append([len(all_old_tokens) - seq_len, steps, compression])

    if return_dict_in_generate:
        pass
    else:
        return input_ids, compression, windows
