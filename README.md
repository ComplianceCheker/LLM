# Replication Package for Compliance Checking 

## Content description

 ```bash
.
├── Code
├── Data
└── Checkpoint

```
        
* Code: implementations of all the elements discussed in the program. 

    * We first use the below LLMs for CauaslLm task.
        * Mistral
        * Zephyr
        * Mixtral 
    * We then use the below LLMs and traditioanl ML algorithms for Bi-Encoders and Embeddigns
        * BERT
        *  SVM and Decision Tree
    * Finally we have the gradio.py which demonstrates the demo based on the uplodade fine-tuned Mixtral model checkpoint

* Data: contains DPA training and test datasets as well as a smaller filtered dataset to test with higher inference speed.
* Checkpoint: The link for fine-tuned Mixtral model


