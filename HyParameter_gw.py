import sys

emb=sys.argv[1]
mini=sys.argv[2]

sh=open('gw_{0}_{1}.sh'.format(emb, mini),'w')

print ('gw_{0}_{1}.sh'.format(emb, mini))
sh.write(
'''
###========================================
#!/bin/bash
#BSUB -n 1
#BSUB -o gw_{0}_{1}.out
#BSUB -e gw_{0}_{1}.err
#BSUB -q "windfall"
#BSUB -J feb_0_gw
#BSUB -R gpu
#---------------------------------------------------------------------
module load cuda/8.0.61
python preprocess.py -train_src /extra/cheny/srctgt/feb_0_gw_train.src \
-train_tgt /extra/cheny/srctgt/feb_0_gw_train.tgt \
-valid_src /extra/cheny/srctgt/feb_0_gw_val.src \
-valid_tgt /extra/cheny/srctgt/feb_0_gw_val.tgt \
-save_data /extra/cheny/feb_0_gw{0}_{1}

python train.py -data /extra/cheny/feb_0_gw{0}_{1} -gpuid 1 \
-src_word_vec_size {0} \
-tgt_word_vec_size {0} \
-batch_size {1} \
-save_model /extra/cheny/feb_0_gw{0}_{1} 

python translate.py -model /extra/cheny/feb_0_gw{0}_{1}_acc*_e13.pt \
-src /extra/cheny/srctgt/feb_0_gw_test.src -output /extra/cheny/feb_0_gw{0}_{1}_pred.txt -replace_unk -verbose

echo $(date) >> hyper_record.txt
echo feb_0_gw{0}_{1} >> hyper_record.txt
perl tools/multi-bleu.perl /extra/cheny/srctgt/feb_0_gw_test.tgt < /extra/cheny/feb_0_gw{0}_{1}_pred.txt >> hyper_record.txt

###end of script
'''.format(emb, mini))

sh.close()
