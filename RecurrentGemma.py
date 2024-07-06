import os
import pathlib
import torch
import sys
import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma
import time
import functools


@functools.cache
def _load(*,model_path,variant) -> recurrentgemma.sampler:
    weights_dir = pathlib.Path(model_path)
    ckpt_path = weights_dir / f'{variant}.pt'
    vocab_path = weights_dir / 'tokenizer.model'
    # Load model weights
    weights_dir = pathlib.Path(model_path)

    # Ensure that the tokenizer is present
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

    # Ensure that the checkpoint is present
    ckpt_path = os.path.join(weights_dir, f'{variant}.pt')
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'


    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1 if '2b' in variant else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1

    # Load parameters
    params = torch.load(str(ckpt_path))
    params = {k: v.to(device=device) for k, v in params.items()}

    # Set up model config
    model_config = recurrentgemma.GriffinConfig.from_torch_params(params, preset=preset)

    # Instantiate the model and load the weights.
    model = recurrentgemma.Griffin(model_config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(params)
    # compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    # Tokenizer
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))

    #Sampler
    return recurrentgemma.Sampler(model=model, vocab=vocab, is_it_model="it" in variant)

@functools.cache
def invoke(sampler,prompt):
    out_data = sampler(input_strings=[prompt], total_generation_steps=30)
    print(out_data.text[0])
    return out_data.text[0]

if __name__ ==  '__main__':
    start_time = time.time()
    variant = '2b'  # @param ['2b', '2b-it', '9b', '9b-it'] {type:"string"}
    model_path = f'models/recurrentgemma-pytorch-2b-v1'
    sampler = _load(model_path=model_path,variant=variant)
    print("--- load model: %s seconds  ---" % (time.time() - start_time))
    response = invoke(sampler, "Task: Suggest names for my 3 cats")
    print("--- load model and response: %s seconds  ---" % (time.time() - start_time))
    print(response)

