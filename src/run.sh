# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25690 -m training.main \
#     --train-data '/tmp/gcc3m/gcc-train-{00000..00331}.tar' \
#     --train-num-samples 2891445 \
#     --model ViT-B-16 \
#     --pretrained laion2b_s34b_b88k \
#     --dataset-type webdataset \
#     --batch-size 512 \
#     --precision amp \
#     --workers 4 \
#     --epochs 10 \

#     #--model ViT-B-32 \
#     #--pretrained laion2b_s34b_b79k \

#     #--model ViT-B-16 \
#     #--pretrained laion2b_s34b_b88k \


# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=25690 -m training.main \
#     --train-data '../data/coco/val2017' \
#       --train_ann_file  '../data/coco/annotations/id_pretrain.json'     --model ViT-B-16     --pretrained laion2b_s34b_b88k     --dataset-type "COCO"    --batch-size 512     --precision amp     --workers 1     --epochs 10

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=25690 -m training.main_rpnClipN     --train-data '../data/coco/val2017'   --train_ann_file  '../data/coco/annotations/id_pretrain.json'      --model ViT-B-16     --pretrained laion2b_s34b_b88k     --dataset-type "COCO_rpnClipN"    --batch-size   32     --precision amp     --workers 1     --epochs 10