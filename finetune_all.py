from datasets import load_from_disk
import torch
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, AutoModelForCausalLM, AutoConfig
from typing import Any
from peft import LoraConfig
from PIL import Image
from trl import SFTConfig, SFTTrainer
# from transformers.utils.device_map import infer_auto_device_map, get_balanced_memory


PROMPT = """
                You are a board certified breast radiologist with lots of experience
                in interpreting screening mammograms. You are meticulous,
                and always provide clear, concise, and clinically actionable reports.

                You are provided with a mammogram image. The image has all 4 breast views shown together.
                The upper two views are the Cranio-Caudal (CC) views of each breast, right and left,
                and the lower two views are the Medio-Lateral Oblique (MLO) views of each breast, right and left.

                At first, you should identify the breast density using the ACR classification,
                which includes:
                - ACR A: Almost entirely fatty
                - ACR B: Scattered fibroglandular densities
                - ACR C: Heterogeneously dense
                - ACR D: Extremely dense

                Then, you should summarize any findings from the images. Mention in which view the findings
                are present. Findings include
                Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry,
                Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening.
                There may be multiple findings in a single image. For normal and healthy breasts, "Healthy Breast. No Findings"

                Finally, assign a BIRADS category based on the findings:
                - BIRADS 1: Negative (no abnormalities)
                - BIRADS 2: Benign (no suspicion of cancer)
                - BIRADS 3: Probably benign (short-term follow-up recommended)
                - BIRADS 4: Suspicious abnormality (biopsy needed)
                - BIRADS 5: Highly suggestive of malignancy (high probability of cancer)

                Also assign the healthy status of the breast.

                Here is the JSON format you should follow for your response:
                {
                    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
                    "findings": "<Summary of any abnormalities as described above>",
                    "birads": "<0|1|2|3|4|5|6> followed by a brief description of the BIRADS category>",
                    "suspicion":"healthy|benign|suspicious"
                }


            """


def format_data(example: dict[str, Any]) -> dict[str, Any]:
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": example["label"],
                },
            ],
        },
    ]
    # print("Formatted example:", example["label"])
    return example



dataset_path = "..."  # Replace this with the dataset dict folder
dataset = load_from_disk(dataset_path)
# row = dataset["train"][0]
# print(row["image"])
# print(row["label"])
data = dataset.map(format_data)

# print(data)



# model_id = "google/medgemma-4b-it"
# model_id = "google/medgemma-1.5-4b-it"
model_id = "google/medgemma-27b-it"
# Check if GPU supports bfloat16
# if torch.cuda.get_device_capability()[0] < 8:
#     raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# model_kwargs = dict(
#     attn_implementation="eager",
#     # torch_dtype=torch.bfloat16,
#     dtype=torch.float16,
#     device_map="auto",
# )

# model_kwargs["quantization_config"] = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # bnb_4bit_compute_dtype=torch.float16,  # ✅ compute in fp16
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=model_kwargs["dtype"],
#     bnb_4bit_quant_storage=model_kwargs["dtype"],
# )

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    # "google/medgemma-4b-it",
    # "google/medgemma-1.5-4b-it",
    "google/medgemma-27b-it",
    quantization_config=bnb,
    device_map="balanced",      # or "balanced"
    dtype="float16",
    low_cpu_mem_usage=True,
)


model.gradient_checkpointing_enable()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Use right padding to avoid issues during training
processor.tokenizer.padding_side = "right"

# print("hf_device_map:", getattr(model, "hf_device_map", None))
# # Inspect a few params
# seen = set()
# for name, p in model.named_parameters():
#     seen.add(str(p.device))
#     if len(seen) >= 2:
#         break
# print("Devices seen among params:", seen)



peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    # r=16,
    r=8,
    bias="none",
    # target_modules="all-linear",
    target_modules=["q_proj","k_proj","v_proj","o_proj"], # adjust if needed
    task_type="CAUSAL_LM",
    # modules_to_save=[
    #     "lm_head",
    #     "embed_tokens",
    # ],
)



def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        img = ex["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        images.append([img])

        chat = processor.apply_chat_template(
            ex["messages"], add_generation_prompt=False, tokenize=False
        ).strip()
        texts.append(chat)

    batch = processor(text=texts, images=images, return_tensors="pt",
                      padding=True, truncation=True, max_length=512)

    labels = batch["input_ids"].clone()

    # Always mask pad
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    # Mask image placeholder tokens (if your template inserts one)
    # Example for a common pattern; adjust to your tokenizer/special token:
    image_token = getattr(processor.tokenizer, "image_token", "<image>")
    if image_token in processor.tokenizer.get_vocab():
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
        labels[labels == image_token_id] = -100
    else:
        # Fallback: try reading from added_special_tokens / attrs, or skip if unknown
        pass

    batch["labels"] = labels
    return batch


num_train_epochs = 1  # @param {type: "number"}
learning_rate = 2e-4  # @param {type: "number"}


# 1) Pick a stable eval set (don't reshuffle each time)
eval_ds = data["test"].select(range(200))  # fixed subset; avoid .shuffle() for stable metrics

# 2) Configure args to track and restore the best checkpoint
args = SFTConfig(
    output_dir="...",  # still needed locally
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    # optim="adamw_torch_fused",
    # optim="adamw_bnb_8bit",
    optim="paged_adamw_8bit",
    logging_steps=50,

    # IMPORTANT: correct key is evaluation_strategy
    eval_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,

    # Keep only 1 checkpoint locally (minimizes disk use)
    save_total_limit=2,

    # Select & restore the best
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",   # or your metric name if you set compute_metrics
    greater_is_better=False,

    # Hub settings — push only once at the end (when best model is loaded)
    push_to_hub=True,
    hub_model_id="...",
    hub_strategy="end",                  # push final state only (= best, thanks to load_best_model_at_end)

    learning_rate=2e-4,
    # bf16=True,
    # fp16=True, 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=eval_ds,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    # compute_metrics=your_compute_metrics_fn,  # uncomment if using a custom metric
    callbacks=[EarlyStoppingCallback(
       early_stopping_patience=10,        # stop if no improvement in 10 evals
       early_stopping_threshold=0.0      # require strictly better (set >0 to require a margin)
    )]
)

trainer.train()
trainer.save_model()
