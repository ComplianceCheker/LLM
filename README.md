# Replication Package for Compliance Checking 

## Content description

 ```bash
.
├── Code
├── Data
├── Paper
└── Checkpoint

```
        
* Code: implementations of all the elements discussed in the program. 

    * We first use the below LLMs for CausalLM task.
        * Mistral
        * Zephyr
        * Mixtral 
    * We then use the below LLMs and traditional ML algorithms for Bi-Encoders and Embeddings
        * BERT
        *  SVM and Decision Tree
    * Finally we have the gradio.py which demonstrates the demo based on the uplodade fine-tuned Mixtral model checkpoint
* Data: contains DPA training and test datasets as well as a smaller filtered dataset to test with higher inference speed.
* Paper: contains the list of papers reviewed throughout the program.
* Checkpoint: The link for fine-tuned Mixtral model


