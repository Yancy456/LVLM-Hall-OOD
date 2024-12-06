conda activate tf4.31
python run_model.py
CUDA_VISIBLE_DEVICES=0 bash ./scripts/Safety/run_LLaVA_7B.sh



bash ./scripts/MAD/run_LLaVA_7B.sh

# jail break dataset
python run_model_old.py --config ./configs/jailbreak.yaml

python run_model_old.py --config ./configs/mad_val.yaml

python run_model_old.py --config ./configs/mad_train.yaml

# jail break dataset
python run_model.py --config ./configs/jailbreak.yaml

# triviaQA
python run_model.py --config ./configs/triviaQA.yaml