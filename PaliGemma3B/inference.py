from PIL import Image
import torch
import fire

from paligemma_processor import ProcessorPaliGemma
from modelling_gemma import PaliGemmaConditionalGeneration, KVCache
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: ProcessorPaliGemma,
    prompt: str,
    image_file_path: str,
    device: str,
):
    # load and process image
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaConditionalGeneration,
    processor: ProcessorPaliGemma,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # generate tokens till we see stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # apply temperature and top_p sampling
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(next_token_logits, top_p=top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # remove batch dims
        generated_tokens.append(next_token)
        # stop if we see stop token
        if next_token.item() == stop_token:
            break
        # append the next token to input_ids and attention_mask for next iteration
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)


def sample_top_p(probs: torch.Tensor, top_p: float):
    # (batch_size, vocab_size)
    probs_sorted, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
    # subtract "sorted_probs" shifts the cumulative sum by one position to the right before masking
    mask = cumulative_probs - probs_sorted > top_p
    probs_sorted[mask] = 0.0  # zeroed out all the probs of tokens that are not selected
    probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))  # renormalize
    # sample from the filtered distribution
    next_token = torch.multinomial(probs_sorted, num_samples=1)
    # map back to original indices
    next_token = torch.gather(sorted_indices, dim=-1, index=next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print(f"Using device: {device}")
    print("Loading model...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_img_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = ProcessorPaliGemma(tokenizer, num_img_tokens, image_size)

    print("Processing prompt...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
