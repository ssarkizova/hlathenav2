#train_file=/Users/cleoforman/PycharmProjects/hlathenav2/notebooks/input/test_training.csv
#val_file=/Users/cleoforman/PycharmProjects/hlathenav2/notebooks/input/test_training.csv
#train_file=/Users/cleoforman/PycharmProjects/hlathenav2/models/data/dummy_features_AAPos_AAPCA_LogTPM_CNN_Kidera_Gene_9/MS_dat_dummy_seed_757_cv0_x10p.pkl.gz_test_merge.txt_slim
#val_file=/Users/cleoforman/PycharmProjects/hlathenav2/models/data/dummy_features_AAPos_AAPCA_LogTPM_CNN_Kidera_Gene_9/MS_dat_dummy_seed_757_cv0_x10p.pkl.gz_valid_merge.txt_slim

train_file=/Users/cleoforman/PycharmProjects/hlathenav2/models/data/test.txt
val_file=/Users/cleoforman/PycharmProjects/hlathenav2/models/data/test.txt

delimiter=' '

#pep_col=pep
#allele_col=mhc
#target_col=tgt
pep_col=seq
allele_col=allele
target_col=label
folds=2
lr=0.01
d=0.1
epochs=1
batch_size=512
prediction_replicates=100
decoy_mul=10
decoy_ratio=1
resample_hits='False'

aa_folder=''

featset=/Users/cleoforman/PycharmProjects/hlathenav2/models/features/basic_features.txt
reps=3

out=../output/test
hla_encoding_file=/Users/cleoforman/PycharmProjects/hlathenav2/hlathena/data/hla_seqs_onehot_clean.csv

#python3 pan_trainer.py [-h] --train_file TRAIN_FILE [--val_file VAL_FILE]
#                      [--pep_col PEP_COL] [--allele_col ALLELE_COL]
#                      [--target_col TARGET_COL] [--fold_col FOLD_COL]
#                      [-kf NUMBER_FOLDS] [-e EPOCHS] [-lr LEARNING_RATE]
#                      [-dr DROPOUT_RATE] [-b BATCH_SIZE] [-pr PRED_REPLICATES]
#                      [-dm DECOY_MUL] [--decoy_ratio DECOY_RATIO]
#                      [-rh RESAMPLE_HITS] [--assign_folds ASSIGN_FOLDS]
#                      [--aa_feature_folder AA_FEATURE_FOLDER]
#                      [--feat_cols FEAT_COLS] [--feat_sets FEAT_SETS]
#                      [--repetitions REPETITIONS] [--seeds SEEDS] [-o OUTDIR]
#                      [-r RUN_NAME]

#echo "python3 pan_trainer.py --train_file $train_file --val_file $val_file --delimiter "$delimiter" --pep_col $pep_col --allele_col $allele_col --target_col $target_col -f $folds -e $epochs -lr $lr -d $d -b $batch_size -pr $prediction_replicates -dm $decoy_mul --decoy_ratio $decoy_ratio --feat_sets $featset --hla_encoding_file $hla_encoding_file --repetitions $reps -o $out --run_name 'test'"

# python3 pan_trainer.py --train_file /Users/cleoforman/PycharmProjects/hlathenav2/models/data/MS_dat_dummy_seed_757_5cv_x10p_train_test_valid.txt --delimiter " " --pep_col seq --allele_col allele --target_col label --fold_col fold -f 5 -e 15 -lr 0.01 -d 0 -b 5000 -pr 10 -dm 10 --decoy_ratio 1000 --resampling_hits --feat_sets /Users/cleoforman/PycharmProjects/hlathenav2/models/features/basic_features.txt --hla_encoding_file /Users/cleoforman/PycharmProjects/hlathenav2/hlathena/data/hla_seqs_PC3_KF10_clean.csv --repetitions 1 -o ../output/test_rh_1000 --run_name test_rh_1000 --aa_feature_folder /Users/cleoforman/PycharmProjects/hlathenav2/models/features/aa_features

# /n/cluster/bin/job_gpu_monitor.sh &
# python3 pan_trainer.py --train_file /home/clf176/Desktop/hlathena_package/hlathenav2/models/data/test_split.txt --delimiter " " --pep_col seq --allele_col allele --target_col label --fold_col fold -f 2 -e 10 -lr 0.01 -d 0 -b 1000 -pr 10 -dm 10 --decoy_ratio 1000 --resampling_hits --feat_sets /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/basic_features.txt --hla_encoding_file /home/clf176/Desktop/hlathena_package/hlathenav2/hlathena/data/hla_seqs_PC3_KF10_clean.csv --repetitions 1 -o /n/scratch/users/c/clf176/test_8workers_seed_904_pan_9mer_rh_1000_gputest --run_name test_seed_904_pan_9mer_rh_1000 --aa_feature_folder /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/aa_features


/n/cluster/bin/job_gpu_monitor.sh &
python3 pan_trainer.py --train_file /home/clf176/Desktop/hlathena_package/hlathenav2/models/data/MS_dat_dummy_seed_757_5cv_x10p_train_test_valid.txt --delimiter " " --pep_col seq --allele_col allele --target_col label --fold_col fold -f 5 -e 15 -lr 0.01 -d 0 -b 5000 -pr 100 -dm 10 --decoy_ratio 1000 --resampling_hits --feat_sets /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/basic_features.txt --hla_encoding_file /home/clf176/Desktop/hlathena_package/hlathenav2/hlathena/data/hla_seqs_PC3_KF10_clean.csv --repetitions 1 -o /n/scratch/users/c/clf176/123_pan_9mer_rh_1000_gput --run_name 123_pan9mer_rh_1000 --aa_feature_folder /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/aa_features


# python3 pan_trainer.py --train_file /home/clf176/Desktop/hlathena_package/hlathenav2/models/data/MS_dat_dummy_seed_757_5cv_x10p_train_test_valid.txt --delimiter " " --pep_col seq --allele_col allele --target_col label --fold_col fold -f 5 -e 15 -lr 0.01 -d 0 -b 5000 -pr 10 -dm 10 --decoy_ratio 1000 --resampling_hits --feat_sets /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/basic_features.txt --hla_encoding_file /home/clf176/Desktop/hlathena_package/hlathenav2/hlathena/data/hla_seqs_PC3_KF10_clean.csv --repetitions 1 -o /n/scratch/users/c/clf176/seed_904_pan_9mer_rh_1000_gputest_numworkers8 --run_name seed_904_pan_9mer_rh_1000 --aa_feature_folder /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/aa_features



# python3 pan_trainer.py --train_file /home/clf176/Desktop/hlathena_package/hlathenav2/models/data/MS_dat_dummy_seed_757_5cv_x10p_train_test_valid.txt --delimiter " " --pep_col seq --allele_col allele --target_col label --fold_col fold -f 5 -e 15 -lr 0.01 -d 0 -b 5000 -pr 10 -dm 10 --decoy_ratio 1000 --resampling_hits --feat_sets /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/basic_features.txt --hla_encoding_file /home/clf176/Desktop/hlathena_package/hlathenav2/hlathena/data/hla_seqs_PC3_KF10_clean.csv --repetitions 1 -o ../output/pan_9mer_rh_1000_gputest --run_name pan_9mer_rh_1000 --aa_feature_folder /home/clf176/Desktop/hlathena_package/hlathenav2/models/features/aa_features

#--train_file $train_file --val_file $val_file --delimiter "$delimiter" --pep_col $pep_col --allele_col $allele_col --target_col $target_col -f $folds -e $epochs -lr $lr -d $d -b $batch_size -pr $prediction_replicates -dm $decoy_mul --decoy_ratio $decoy_ratio --feat_sets $featset --hla_encoding_file $hla_encoding_file --repetitions $reps -o $out --run_name 'test'