# Exp 01

## Environment Setup

1. Create conda environment and install pytorch
    ```
    conda create -n exp01 python=3.10
    conda activate exp01
    pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    ```

2. Make sure nvcc can be found.
    ```shell
    nvcc --version
    ```
    If it is not installed, install it by following the instruction [here](https://developer.nvidia.com/cuda-downloads)  
    Or you can try `bash setup_nvcc.sh`.    

3. Follow the [readme](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to install flash-attention.
And you can try `bash setup_flash-attn.sh`.

    If you encounter some errors when running train scripts, try to install `flash-attn` from source code:  
    
    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    pip install . --no-build-isolation
    ```

4. `pip install -r requirements.txt`

## Training

```shell
bash run_train_single_node.sh
```

# Exp 05
Train Llama-3-70B with 16*80GB A100 GPUs using accelerate  

## Setup environment
Follow the README.md in exp01 to setup the environment.  

## Train
1. Download and process alpaca dataset in each node.  

    You should run the following command in each node since data will be transferred to each node automatically.  

    Or make sure you have a shared disk to store the dataset.  
    ```bash
    python process_alpaca.py
    python data_filter.py
    ```

2. Update deepspeed config files `ds_node1_config.yaml` and `ds_node2_config.yaml` with the correct `main_process_ip`, which is the IP address of the node 1.  

3. Make sure you have the write access to the `OUTPUT_DIR` in `run_ds_node1.sh` and `run_ds_node2.sh`, or you can change the `OUTPUT_DIR`.  

### Multi Node training
4. Start Training  

    In Node 1,  
    ```bash
    bash run_ds_node1.sh
    ```

    In Node 2,  

    ```bash
    bash run_ds_node2.sh
    ```

5. Copy the deepspeed model states to a single node and transfer the model states to fp32 pytorch.bin file.  
You should mannually copy the model states from node 2 to node 1 using `scp` or other tools. The process may cost serval hours so a shared disk is recommended.  
And then run `python zero_to_fp32.py` in the `OUTPUT_DIR`.  


### Fine Tune 8B-Instruct Model
We use DeepSpeed Zero-2 to fine-tune the 8B-Instruct model.  

```
bash train_8b.sh
```

## Generate Response
~~### Transformers Inference~~

**`generate.py` is too slow to use. We should use vllm for inference.**

```
python generate.py <Your ckpt path>
pip install alpaca-eval
export OPENAI_API_KEY=<your_openai_api_key>
alpaca_eval --model_outputs alpaca_eval_70b.json
```

### Use vLLM for inference

```
# Latest version may have some bugs, so we use v0.5.2
# Start vllm server
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.2/vllm-0.5.2-cp310-cp310-manylinux1_x86_64.whl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MODEL_DIR=alpaca_70b_ckpt_step277
vllm serve $MODEL_DIR --tensor-parallel-size 4 --chat-template ./template_chatml.jinja


# client
# You should change the VLLM_MODEL_NAME to the ${MODEL_DIR}
export MODEL_DIR=alpaca_70b_ckpt_step277
python vllm_client.py result.json
```
It may cost **three minutes** to load the 70B model.

You may meet the error: `ModuleNotFoundError: No module named "vllm._C" `, which can be fixed by the [issue](https://github.com/vllm-project/vllm/issues/1814#issuecomment-1837122930)


## Get alpaca-eval score

```
pip install alpaca-eval
export OPENAI_API_KEY=<your_openai_api_key>
alpaca_eval --model_outputs result.json
```

