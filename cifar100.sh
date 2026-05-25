ARCH="resnet32_quant"
DATASET="cifar100"
BATCH_SIZE=512
ENSEMBLE_BY="seed"
ENSEMBLE_METHOD="UniformEnsembling"

TOTAL_EPOCHS=400 
N_EPOCHS=400 
N_EPOCHS_PER_PHASE=30 
N_PHASES_QMS=1
N_EPOCH_SOUP_AFTER_FINETUNE=0

W_BIT=4
A_BIT=4
WEIGHT_LEVELS=$(( 2 ** W_BIT ))  
ACT_LEVELS=$(( 2 ** A_BIT ))      


N_SPLITS_TOTAL_QMS=30
UPDATE_QUANT_QMS=True
UPDATE_PRUNING_QMS=False
FREEZE_BATCHNORM_BOOL_QMS=False


for run_id in; do
    python3 main.py --cifar100_pretrained --arch resnet18_tiny --dataset tinyimagenet  --batch_size 1024 --n_epochs 100 --strategy Dense --run_id ${run_id} --total_epochs 100  --w_bit 32 --a_bit 32
    python3 main.py --cifar100_qat --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --run_id "${run_id}" --strategy WarmupQAT --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --update_quant "True" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
    for bb in 0.00 ; do 
        for sp in  0.0; do 
            for p in 1 ; do 
                for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30   ; do
                    python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --split_val "${k}" --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy QAT --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
                done
                python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy Ensemble --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
            done
        done
    done
done

UPDATE_QUANT_QMS=False
UPDATE_PRUNING_QMS=True
FREEZE_BATCHNORM_BOOL_QMS=False

for run_id in   ; do
    for bb in 0.35 ; do 
        for sp in  0.05; do 
            for p in 1 ; do 
                for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30   ; do
                    python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --split_val "${k}" --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy QAT --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
                done
                python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy Ensemble --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
            done
        done
    done
done

UPDATE_QUANT_QMS=False
UPDATE_PRUNING_QMS=True
FREEZE_BATCHNORM_BOOL_QMS=False



for run_id in   ; do
    for bb in 0.4 ; do 
        for sp in  0.01; do 
            for p in 1 ; do 
                for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30   ; do
                    python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --split_val "${k}" --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy QAT --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
                done
                python3 main.py --cifar100_qms --arch "${ARCH}" --dataset "${DATASET}" --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS} --total_epochs ${TOTAL_EPOCHS} --phase "${p}" --n_phases ${N_PHASES_QMS} --n_splits_total ${N_SPLITS_TOTAL_QMS} --run_id "${run_id}" --strategy Ensemble --ensemble_by "${ENSEMBLE_BY}" --ensemble_method "${ENSEMBLE_METHOD}" --pruning_ratio "${sp}" --n_epochs_per_phase ${N_EPOCHS_PER_PHASE} --n_epoch_soup_after_finetune ${N_EPOCH_SOUP_AFTER_FINETUNE} --w_bit ${W_BIT} --a_bit ${A_BIT} --weight_levels ${WEIGHT_LEVELS} --act_levels ${ACT_LEVELS} --boundaryRange "${bb}" --update_quant "${UPDATE_QUANT_QMS}" --update_pruning "${UPDATE_PRUNING_QMS}" --freeze_batchnorm_bool "${FREEZE_BATCHNORM_BOOL_QMS}"
            done
        done
    done
done
