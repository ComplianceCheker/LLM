from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, GenerationConfig, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, LoraConfig, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
import evaluate
import nltk
from trl import SFTTrainer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from auto_gptq import exllama_set_max_input_length
import os
import gradio as gr
import torch
# torch.cuda.empty_cache()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

#Checkpoints

coder_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"


"""#MODEL LOADING"""

tokenizer = AutoTokenizer.from_pretrained(coder_checkpoint)
tokenizer.pad_token = tokenizer.unk_token
#tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(coder_checkpoint, quantization_config=bnb_config, device_map='auto')

peft_model = PeftModel.from_pretrained(model,
                                       'Mixtral_CODv1/checkpoint-3540',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)


#INFERENCE

def inference(sentences):
    
    prompt_temp = f"""You are a legal expert, your task to take in Policies and text, to identify which policy the text was applicable to, if there is no direct connection, just output R99.
Policies: 
R1 - The DPA shall contain at least one controller's identity and contact details.
R2 - The DPA shall contain at least one processor's identity and contact details.
R3 - The DPA shall contain the duration of the processing.  (Art. 28(3))
R4 - The DPA shall contain the nature and purpose of the processing. (Art. 28(3))
R5 - The DPA shall contain the types of personal data. (Art. 28(3))
R6 - The DPA shall contain the categories of data subjects. (Art. 28(3))
R7 - The  organizational and technical measures to ensure a level of security can include: (a) pseudonymisation and encryption of personal data, (b) ensure confidentiality, integrity, availability and resilience of processing systems and services, (c) restore the availability and access to personal data in a timely manner in the event of a physical or technical incident, and (d) regularly testing, assessing and evaluating the effectiveness of technical and organisational measures for ensuring the security of the processing. (Art. 32(1))
R8 - The notification of personal data breach shall at least include (a) the nature of personal data breach; (b) the name and contact details of the data protection officer; (c) the consequences of the breach; (d) the measures taken or proposed to mitigate its effects. (Art. 33(3))
R10 - The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.  (Art. 28(2))
R11 - In case of general written authorization, the processor shall inform the controller of any intended changes concerning the addition or replacement of sub-processors. (Art. 28(2))
R12 - The processor shall process personal data on documented instructions from the controller.  (Art. 28(3a))
R15 - The processor shall ensure that persons authorized to process personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality. (Art. 28(3b))
R16 - The processor shall take all measures required pursuant to Article 32 or to ensure the security of processing. (Art. 28(3c))
R17 - The processor shall assist the controller in fulfilling its obligation to respond to requests for exercising the data subject's rights. (Art. 28(3e))
R22 - The processor shall assist the controller in ensuring compliance with the obligations pursuant to data protection impact assessment (DPIA). (Art. 28(3f), Art.35)
R23 - The processor shall return or delete all personal data to the controller after the end of the provision of services relating to processing. (Art. 28(3g))
R24 - The processor shall immediately inform the controller if an instruction infringes the GDPR or other data protection provisions.  (Art. 28(3h))
R25 - The processor shall make available to the controller information necessary to demonstrate compliance with the obligations Article 28 in GDPR. (Art. 28(3h))
R26 - The processor shall allow for and contribute to audits, including inspections, conducted by the controller or another auditor mandated by the controller. (Art. 28(3h))
R27 - The processor shall impose the same obligations referred to in Article 28(3) in GDPR on the engaged sub-processors by way of contract or other legal act under Union or Member State law. (Art. 28(4))
R28 - The processor shall remain fully liable to the controller for the performance of sub-processor's obligations. (Art. 28(4))
R30 - The processor shall not transfer personal data to a third country or international organization without  a prior specific or general authorization of the controller. (Art. 28(3a))
R34 - The processor shall notify the controller without undue delay after becoming aware of a personal data breach. (Art. 33(2))
R36 - In case of general written authorization, the controller shall have the right to object to changes concerning  the addition or replacement of sub-processors, after having been informed of such intended changes by the processor. (Art. 28(2))
R38 - The controller shall have the right to terminate the DPA in certain cases.

--- example 1 start ---
Text: [NAME ACCOUNTANCY PRACTICE], with its registered office in [city, street and house number], hereinafter referred to as: "Processor", "We", "Us" or "Our", duly represented in this matter by [name + position];
Prediction: R2
--- example 1 end ---

--- example 2 start ---
Text: The Processing will be carried out in accordance with Your written instructions, unless We are obliged by law or regulations to act differently (for example, when considering whether an "unusual transaction" should be reported within the context of the Money Laundering and Terrorist Financing Prevention Act (Wwft)).
Prediction: R12
--- example 2 end ---
"""

    prompt = f"[INST]{prompt_temp} Sentence:{sentences}[/INST] Answer:"
    model_inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = peft_model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    split_text_p = text.split("Answer:")
    return split_text_p[1].strip() if len(split_text_p) > 1 else "No output"

# Create Gradio Interface
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(placeholder="Enter your DPA Sentence here", label="Compliance Checking", lines=2),
    ],
    outputs=gr.Textbox(placeholder="Generated code will appear here"),
    title="COMPLIANCE CHECKER KNOWD",
    description="Enter the DPA Senetence to check the compliant policy number.",
    theme="huggingface",
)
# Launch the Gradio Interface
iface.launch(share=True)