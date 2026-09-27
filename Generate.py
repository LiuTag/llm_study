import torch
from BPE import BPETokenizer
from Attention import softmax


def generate_text(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> str:
    if prompt == "":
        raise ValueError("输入提示词不能为空")
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p应该在(0,1]之间")
    if context_length <= 0:
        raise ValueError("context_length应该为正整数")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens应该在大于等于0")
    if temperature <= 0:
        raise ValueError("temperature应该为正数")

    
    
    prompt_token_list = tokenizer.encode(prompt)

    model_training = model.training
    if model_training:
        model.eval()

    try:
        generate_token_count = 0
        while generate_token_count < max_new_tokens:
            input_list = prompt_token_list if len(prompt_token_list) <= context_length else prompt_token_list[len(prompt_token_list)-context_length:len(prompt_token_list)]
            input_tokens = torch.atleast_2d(torch.tensor(input_list,device=device,dtype=torch.int64))

            with torch.no_grad():
                logits = model(input_tokens)
            logit = logits[0][-1]
            p = softmax(logit/temperature,0)
            values, indices = torch.sort(p,descending=True)


            p_count = 0
            p_len = 0
            for i in values:
                if p_count >= top_p:
                    break
                else:
                    p_count += i
                    p_len += 1

            values = values[0:p_len] / p_count
            indices = indices[0:len(values)]

            idx = torch.multinomial(values, num_samples=1, replacement=True)
            value = indices[idx].tolist()
            if value[0] == eos_token_id:
                break
            else:
                prompt_token_list += value
                generate_token_count += 1
    finally:
        if model_training:
            model.train()

    return tokenizer.decode(prompt_token_list)
    

    

