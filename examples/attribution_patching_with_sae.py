import torch
from transformer_lens import ActivationCache, HookedSAETransformer
from circuit_finder.pretrained import load_attn_saes
from circuit_finder.constants import device
from transformer_lens import utils
from circuit_finder.utils import clear_memory
from circuit_finder.patching.forward_backward_cache import ForwardBackwardCache

filter_sae_acts = lambda name: ("hook_sae_acts_post" in name)


def get_cache_fwd_and_bwd(model, tokens, metric):
    with ForwardBackwardCache(model, filter_sae_acts) as cache:
        value = metric(model(tokens))
        value.backward()

    return (
        value.item(),
        cache.act_cache,
        cache.grad_cache,
    )


def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


def attr_patch_sae_acts(
    clean_cache: ActivationCache,
    clean_grad_cache: ActivationCache,
    site: str,
    layer: int,
):
    clean_sae_acts_post = clean_cache[
        utils.get_act_name(site, layer) + ".hook_sae_acts_post"
    ]
    clean_grad_sae_acts_post = clean_grad_cache[
        utils.get_act_name(site, layer) + ".hook_sae_acts_post"
    ]
    sae_act_attr = clean_grad_sae_acts_post * (0 - clean_sae_acts_post)
    return sae_act_attr


if __name__ == "__main__":
    # Load the model
    l5_sae = load_attn_saes([5])[5]
    model = HookedSAETransformer.from_pretrained("gpt2")

    # Load the prompts
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    names = [
        (
            " John",
            " Mary",
        ),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]
    # List of prompts
    prompts = []
    # List of answers, in the format (correct, incorrect)
    answers = []
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(prompt_format)):
        for j in range(2):
            answers.append((names[i][j], names[i][1 - j]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(prompt_format[i].format(answers[-1][1]))
    answer_tokens = torch.tensor(answer_tokens).to(device)

    # Get the original logits
    tokens = model.to_tokens(prompts, prepend_bos=True)
    original_logits, cache = model.run_with_cache(tokens)
    original_per_prompt_logit_diff = logits_to_ave_logit_diff(
        original_logits, answer_tokens, per_prompt=True
    )
    BASELINE = original_per_prompt_logit_diff

    del cache
    clear_memory()

    # Define the IOI metric
    def ioi_metric(logits, answer_tokens=answer_tokens):
        return (
            logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True) - BASELINE
        ).sum()

    clean_tokens = tokens.clone()

    # Attribution patching by splicing the SAE into the model.
    with model.saes([l5_sae]):
        with ForwardBackwardCache(model, filter_sae_acts) as cache:
            clean_value = ioi_metric(model(tokens))
            clean_value.backward()

    clean_cache = cache.act_cache
    clean_grad_cache = cache.grad_cache

    print("Clean Value:", clean_value.item())
    print("Clean Activations Cached:", len(clean_cache))
    print("Clean Gradients Cached:", len(clean_grad_cache))

    site = "z"
    layer = 5
    sae_act_attr = attr_patch_sae_acts(clean_cache, clean_grad_cache, site, layer)
    print("SAE Act Attr:", sae_act_attr.shape)
