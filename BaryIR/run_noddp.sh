# python trainer_bary_noddp.py --batchSize=3 --nEpochs=57 --pairnum=10000000 --Sigma=10000 --sigma=1 --de_type derain dehaze lowlight denoise_25 deblur --patch_size=128 --type all --gpus=5 --backbone=BaryNet --step=31 --resume=checkpoint/model_allBaryNet128__57_1.0.pth --num_source=5

python tester_bary.py