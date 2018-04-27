import sys

emb=sys.argv[1]
mini=sys.argv[2]

sh=open('parapart_{0}_{1}.sh'.format(emb, mini),'w')

print ('parapart_{0}_{1}.sh'.format(emb, mini))
sh.write(
'''
###========================================
#!/bin/bash
#BSUB -n 1
#BSUB -o parapart{0}_{1}.out
#BSUB -e parapart{0}_{1}.err
#BSUB -q "windfall"
#BSUB -J parapart{0}_{1}
#BSUB -R gpu
#---------------------------------------------------------------------
module load cuda/8.0.61
python preprocess.py \
-train_src /extra/cheny/srctgt/feb_0_multi_partial_train_complete.src \
-train_tgt /extra/cheny/srctgt/feb_0_multi_partial_train_complete.tgt \
-valid_src /extra/cheny/srctgt/feb_0_multi_partial_val_complete.src \
-valid_tgt /extra/cheny/srctgt/feb_0_multi_partial_val_complete.tgt \
-save_data /extra/cheny/0_multi_partial_complete{0}_{1}

python train.py -data /extra/cheny/0_multi_partial_complete{0}_{1} \
-src_word_vec_size {0} \
-tgt_word_vec_size {0} \
-batch_size {1} \
-save_model /extra/cheny/0_multi_partial_complete{0}_{1} -gpuid 1

python translate.py -model /extra/cheny/0_multi_partial_complete{0}_{1}*_e13.pt \
-src /extra/cheny/srctgt/feb_0_gd_test.src \
-output /extra/cheny/0_multi_partial_complete_pred{0}_{1}.txt -replace_unk -verbose

echo $(date) >> hyper_record.txt
echo 0_multi_partial_complete{0}_{1} >> hyper_record.txt
perl tools/multi-bleu.perl /extra/cheny/srctgt/feb_0_gd_test.tgt \
< /extra/cheny/0_multi_partial_complete_pred{0}_{1}.txt >> hyper_record.txt
'''.format(emb, mini))

sh.close()
