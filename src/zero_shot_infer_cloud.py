import os
import copy
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10 as cifar_10
from torchvision.datasets import CIFAR100 as cifar_100
import torchvision.datasets as dset
from classification import ViT_Classifier, load_model
from tuning_util import maybe_dictionarize, Places, Textures, ImageNet, iNaturalist, SUN,COCO_val,COCO_val_cloud,maybe_dictionarize4cloud
from tuning_cfg import parse_arguments
from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
from scipy.special import logsumexp
import numpy as np
import pandas as pd
import shutil
from tuning_util import COCODataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
from open_clip.model import get_valid_cropped_images_texts,labels_to_descriptions_COCO

to_np = lambda x: x.detach().cpu().numpy()
def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0]) # maximum value of the logits from the final neural network layer before any normalization like softmax
def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0]) # maximum probability value after applying the softmax function to the logits:
def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1)) #  logarithm of the sum of exponentials of input elements where lower energy values can indicate higher confidence

def infer(args, pth_dir, epoch, model_type='ViT-B-16',vis=True):
    pth_name = os.path.join("checkpoints", "epoch_" + str(epoch) + ".pt")
    pre_train = os.path.join(pth_dir, pth_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 512
    train_data = "imagenet"
    # dataset = ImageNet()
    dataset=COCODataset(img_dir="/home/ywan1084/Documents/Github/cloud/data/coco/val2017",annotation_file="/home/ywan1084/Documents/Github/cloud/data/coco/annotations/id_pretrain.json",is_train=False, transform=None)
    
    # TODO:get rpn, backbone, load weights, 

    def create_rpn():
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),  # Typical scales for P2 to P6
            aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Assuming the same aspect ratios for all levels
        )
        
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=RPNHead(256, anchor_generator.num_anchors_per_location()[0]),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )
        return rpn
    rpn_model= create_rpn().to(device)

    resnet_model = fasterrcnn_resnet50_fpn(pretrained=True)
    backbone = resnet_model.backbone.to(device)
    checkpoint = torch.load(pre_train, map_location=device)
    rpn_model.load_state_dict(checkpoint['rpn'])
    
    vit_class, process_train, process_test = load_model(model_type=model_type, pre_train=pre_train, dataset=dataset, device=device)
    # vit_class, process_train, process_test = load_model(model_type=model_type, pre_train=pre_train, dataset=None, device=device)


    vit_class.fc_yes.requires_grad = False
    vit_class.fc_no.requires_grad = False
    
    print('Model loaded!')

    # if train_data == "imagenet":
    #     dataset = ImageNet(preprocess_train = process_train, preprocess_test = process_test, batch_size = batch_size)
    #     test_dataset = {
    #         "iNaturalist": iNaturalist(preprocess_test = process_test, batch_size = batch_size).test_loader,
    #         "SUN": SUN(preprocess_test = process_test, batch_size = batch_size).test_loader,
    #         "Textures": Textures(preprocess_test = process_test, batch_size = batch_size).test_loader,
    #         "Places": Places(preprocess_test = process_test, batch_size = batch_size).test_loader,
    #     }
    if train_data == "imagenet":
        dataset = COCO_val_cloud(img_dir='/home/ywan1084/Documents/Github/cloud/data/coco/val2017',
                           annotation_file='/home/ywan1084/Documents/Github/cloud/data/coco/annotations/id_test.json',preprocess_test = process_test,
                           batch_size = batch_size,)
        test_dataset = {
            "COCO_ood": COCO_val_cloud(img_dir='/home/ywan1084/Documents/Github/cloud/data/coco/val2017',
                             annotation_file='/home/ywan1084/Documents/Github/cloud/data/coco/annotations/ood_test.json',preprocess_test = process_test,
                             batch_size = batch_size,
                             OOD=True).test_loader,
        }
    
    print('Dataset processed!')
    test_loader = dataset.test_loader
    # test_loader = dataset

    vit_class = vit_class.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    backbone = torch.nn.DataParallel(backbone, device_ids=devices)
    rpn_model = torch.nn.DataParallel(rpn_model, device_ids=devices)
    vit_class = torch.nn.DataParallel(vit_class, device_ids=devices)
   
    id_lis_epoch, ood_lis_epoch = cal_all_metric(pth_dir,test_loader, backbone,rpn_model,vit_class, epoch, test_dataset)
    # id_lis_epoch, ood_lis_epoch = cal_all_metric(test_dataset, model, epoch, test_dataset)
    if vis:
        visualize(pth_dir,test_loader, backbone,rpn_model,vit_class, epoch, test_dataset)
    
    return ood_lis_epoch
    
#TODO: the forward pass into rpn, then clipn(vit_class).
def cal_all_metric(pth_dir,id_dataset, backbone,rpn,vit_class, epoch, ood_dataset=None, flag = True):
    backbone.eval()
    rpn.eval()
    vit_class.eval()
    pred_lis = []
    gt_lis = []
    top5_pred_lis=[]

    # for i, batch in enumerate(dataloader):
    #     step = num_batches_per_epoch * epoch + i
    #     scheduler(step)

    #     images, bboxes_info = batch
    #     bboxes=bboxes_info['bboxes'],
    #     texts=bboxes_info['texts']
    #     bbox_labels=bboxes_info['bbox_labels']
    #     original_images=bboxes_info['original_images']
        
    #     images = images.to(device=device, non_blocking=True)
    #     # texts = texts.to(device=device, non_blocking=True)
    #     bbox_targets=[]
        
    #     for labels, boxes in zip(bbox_labels, bboxes[0]):
    #         target_dict = {
    #             'boxes': boxes,  # 如果有多个框，应该是torch.stack(boxes)
    #             'labels': torch.tensor(labels)
    #         }
    #         bbox_targets.append(target_dict)
        
    #     data_time_m.update(time.time() - end)
    #     optimizer.zero_grad()

    #     with autocast():
    #         image_features, text_features, text_features_no, logit_scale,proposal_loss = model(images, texts,original_images,device,bbox_targets,)
    #         loss_bin_yes, loss_bin_no, loss_tso = loss(image_features, text_features, text_features_no, logit_scale) #+ loss(image_features, text_features_no, logit_scale)
    #         objectness_loss,rpn_box_reg_loss=proposal_loss['loss_objectness'],proposal_loss['loss_rpn_box_reg']
    #         lam=1.0
    #         # total_loss = (loss_bin_yes + loss_bin_no + loss_tso) * 0.5+(lam*objectness_loss+rpn_box_reg_loss)*0.5  # default: 0.5
    #         total_loss = (loss_bin_yes + loss_bin_no + loss_tso+lam*objectness_loss+rpn_box_reg_loss) * 0.1
    #     #time.sleep(1000)
    

    ind_logits, ind_prob, ind_energy = [], [], []
    if flag:
        ind_ctw, ind_atd = [], []
    res = []

    # TODO:use train logic
    json_data = []
    with torch.no_grad():
        image_id = 0
        for i, batch in tqdm(enumerate(id_dataset)):
            # batch = maybe_dictionarize4cloud(batch)
            # print(i,'batch size:',len(batch['labels']))
            # inputs = batch["images"].cuda()
            # labels = batch['labels'].cuda()
            # logits, logits_no, _ = clipn(inputs)

            images, bboxes_info = batch
            # images, bboxes_info = batch['image'],batch['bboxes_info']
            #images: torch.Size([512, 3, 224, 224])
            bboxes=bboxes_info['bboxes'] #tensor list of coordinates: (512,k,4)
            texts=bboxes_info['texts'] # list, (512,k)
            bbox_labels=bboxes_info['bbox_labels'] # list, (512,k)
            original_images=bboxes_info['original_images'] # list,(512,PIL with original size)
            # print(type(images),type(bboxes_info))
            # print(images.shape)
            # print(len(bboxes))
            # print(len(texts))
            # print(len(bbox_labels))
            # print(len(original_images))
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            images = images.to(device=device, non_blocking=True)
            # texts = texts.to(device=device, non_blocking=True)
            bbox_targets=[]
            
            # for labels, boxes in zip(bbox_labels, bboxes[0]):
            #     target_dict = {
            #         'boxes': boxes,  # 如果有多个框，应该是torch.stack(boxes)
            #         'labels': torch.tensor(labels)
            #     }
            #     bbox_targets.append(target_dict)
            for labels, boxes in zip(bbox_labels, bboxes):
                target_dict = {
                    'boxes': boxes,
                    'labels': torch.tensor(labels)
                }
                bbox_targets.append(target_dict)
            # print(len(bbox_targets))
            # print(bbox_targets[0]['boxes'])
            # print(bbox_targets[0]['labels'])
            #forward
            bbox_targets = [{k: v.to(device) for k, v in t.items()} for t in bbox_targets]
            backbone_features=backbone(images)
            height, width = images.shape[2], images.shape[3]
            image_sizes = [(height, width)] * images.shape[0] 
            images_list = ImageList(images, image_sizes)
            # backbone_features = F.normalize(backbone_features, dim=-1)
            proposals, proposal_losses = rpn(images_list, backbone_features, targets=bbox_targets)
            #print('num of proposals',len(proposals)) #list,512
            valid_cropped_images,valid_labels,gt_cropped_images,gt_labels,valid_proposals=get_valid_cropped_images_texts(images,proposals,original_images,bbox_targets,IoU_threshold=0.5)
            print('num of proposals',len(proposals))
            print(proposals[1])
            print('valid proposals',len(valid_proposals))
            print(valid_proposals[1])
            print('valid labels',len(valid_labels))
            print('valid images',len(valid_cropped_images))
            # resize_transform = transforms.Resize((224, 224))
            # valid_cropped_images= [resize_transform(img) for img in valid_cropped_images]
            valid_cropped_images = torch.stack(valid_cropped_images)
            valid_cropped_images=valid_cropped_images.to(device)
            # gt_cropped_images=[resize_transform(img) for img in gt_cropped_images]
            # gt_cropped_images=torch.stack(gt_cropped_images)
            # gt_cropped_images=gt_cropped_images.to(device)
            
            
            valid_descriptions = labels_to_descriptions_COCO(valid_labels)
            from open_clip import tokenize
            valid_texts=[tokenize(text) for text in valid_descriptions]
            valid_texts=torch.stack(valid_texts).to(device)
            
            # gt_descriptions = labels_to_descriptions_COCO(gt_labels)
            # from open_clip import tokenize
            # gt_texts=[tokenize(text) for text in gt_descriptions]
            # gt_texts=torch.stack(gt_texts).to(device)
            
            
            # valid_cropped_images=torch.cat((valid_cropped_images, gt_cropped_images), dim=0)
            # valid_texts=torch.cat((valid_texts, gt_texts), dim=0)
            
            # image_features, text_features, text_features_no,logit_scale=self.clip_model(valid_cropped_images,valid_texts)
            #print(len(valid_cropped_images))
            logits, logits_no, _=vit_class(valid_cropped_images)
            
            
            
            batch_pred = list(torch.argmax(logits, -1).detach().cpu().numpy())
            pred_lis += batch_pred
            # labels
            labels = torch.tensor(valid_labels)
            gt_lis += list(labels.detach().cpu().numpy())

            top5_pred = torch.topk(logits, 5, dim=-1)[1].detach().cpu().numpy()
            top5_pred_lis += list(top5_pred)

            
            ind_logits += list(max_logit_score(logits))
            ind_prob += list(msp_score(logits))
            ind_energy += list(energy_score(logits))
            def top_k_accuracy(gt, pred, k=5):
                correct = 0
                for i in range(len(gt)):
                    if gt[i] in pred[i][:k]:
                        correct += 1
                return correct / len(gt)
            
            if flag:
                idex = torch.argmax(logits, -1).unsqueeze(-1)
                yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                yesno = torch.softmax(yesno, dim=-1)[:,:,0] # all pij(yes)
                yesno_s = torch.gather(yesno, dim=1, index=idex) #pij(yes) where j is the argmax
                ind_ctw += list(yesno_s.detach().cpu().numpy()) # larger than 0.5 then the decision is ID, vice versa
                batch_ind_atd = list((yesno * torch.softmax(logits, -1)).sum(1).detach().cpu().numpy())
                ind_atd += batch_ind_atd

            bbox_index=0
            for _, bboxes in enumerate(valid_proposals):
                for bbox in bboxes:
                    json_entry = {
                        "image_id": image_id,
                        "category_id": valid_labels[bbox_index],
                        "bbox": bbox,
                        "score": ind_ctw[bbox_index]
                    }
                    json_data.append(json_entry)
                    bbox_index += 1
                image_id+=1



        for name, ood_data in ood_dataset.items():
            ood_logits, ood_prob, ood_energy = [], [], []
            if flag:
                ood_ctw, ood_atd = [], []
            for i, batch in tqdm(enumerate(ood_data)):
                images, bboxes_info = batch
                # images, bboxes_info = batch['image'],batch['bboxes_info']
                #images: torch.Size([512, 3, 224, 224])
                bboxes=bboxes_info['bboxes'] #tensor list of coordinates: (512,k,4)
                texts=bboxes_info['texts'] # list, (512,k)
                bbox_labels=bboxes_info['bbox_labels'] # list, (512,k)
                original_images=bboxes_info['original_images'] # list,(512,PIL with original size)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                images = images.to(device=device, non_blocking=True)
                # texts = texts.to(device=device, non_blocking=True)
                bbox_targets=[]
                
                # for labels, boxes in zip(bbox_labels, bboxes[0]):
                #     target_dict = {
                #         'boxes': boxes,  # 如果有多个框，应该是torch.stack(boxes)
                #         'labels': torch.tensor(labels)
                #     }
                #     bbox_targets.append(target_dict)
                for labels, boxes in zip(bbox_labels, bboxes):
                    target_dict = {
                        'boxes': boxes,
                        'labels': torch.tensor(labels)
                    }
                    bbox_targets.append(target_dict)
            

                #forward
                bbox_targets = [{k: v.to(device) for k, v in t.items()} for t in bbox_targets]
                backbone_features=backbone(images)
                height, width = images.shape[2], images.shape[3]
                image_sizes = [(height, width)] * images.shape[0] 
                images_list = ImageList(images, image_sizes)
                # backbone_features = F.normalize(backbone_features, dim=-1)
                proposals, proposal_losses = rpn(images_list, backbone_features, targets=bbox_targets)
                valid_cropped_images,valid_labels,gt_cropped_images,gt_labels,valid_proposals=get_valid_cropped_images_texts(images,proposals,original_images,bbox_targets,IoU_threshold=0.5)
                # resize_transform = transforms.Resize((224, 224))
                # valid_cropped_images= [resize_transform(img) for img in valid_cropped_images]
                valid_cropped_images = torch.stack(valid_cropped_images)
                valid_cropped_images=valid_cropped_images.to(device)
                # gt_cropped_images=[resize_transform(img) for img in gt_cropped_images]
                # gt_cropped_images=torch.stack(gt_cropped_images)
                # gt_cropped_images=gt_cropped_images.to(device)
                
                
                valid_descriptions = labels_to_descriptions_COCO(valid_labels)
                from open_clip import tokenize
                valid_texts=[tokenize(text) for text in valid_descriptions]
                valid_texts=torch.stack(valid_texts).to(device)
                
                # gt_descriptions = labels_to_descriptions_COCO(gt_labels)
                # from open_clip import tokenize
                # gt_texts=[tokenize(text) for text in gt_descriptions]
                # gt_texts=torch.stack(gt_texts).to(device)
                
                
                # valid_cropped_images=torch.cat((valid_cropped_images, gt_cropped_images), dim=0)
                # valid_texts=torch.cat((valid_texts, gt_texts), dim=0)
                
                # image_features, text_features, text_features_no,logit_scale=self.clip_model(valid_cropped_images,valid_texts)
                logits, logits_no, _=vit_class(valid_cropped_images)

                ood_logits += list(max_logit_score(logits))
                ood_prob += list(msp_score(logits))
                ood_energy += list(energy_score(logits))
            
                if flag:
                    idex = torch.argmax(logits, -1).unsqueeze(-1) #j
                    yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                    yesno = torch.softmax(yesno, dim=-1)[:,:,0]
                    yesno_s = torch.gather(yesno, dim=1, index=idex)

                    ood_ctw += list(yesno_s.detach().cpu().numpy())
                    batch_ood_atd = list((yesno * torch.softmax(logits, -1) ).sum(1).detach().cpu().numpy())
                    ood_atd += batch_ood_atd
                
                for i in range(0,len(valid_proposals)):
                    if len(valid_proposals[i])!=0:
                        print(i)
                bbox_index=0
                for _, bboxes in enumerate(valid_proposals):
                    for bbox in bboxes:
                        json_entry = {
                            "image_id": image_id,
                            "category_id": valid_labels[bbox_index],
                            "bbox": bbox,
                            "score": ood_ctw[bbox_index]
                        }
                        json_data.append(json_entry)
                        bbox_index += 1
                    image_id+=1
        

    # with torch.no_grad():
    #     for i, batch in tqdm(enumerate(id_dataset)):
            
            # batch = maybe_dictionarize(batch)
    #         print(i,'batch size:',len(batch['labels']))
    #         inputs = batch["images"].cuda()
    #         labels = batch['labels'].cuda()
    #         logits, logits_no, _ = model(inputs)
            
    #         batch_pred = list(torch.argmax(logits, -1).detach().cpu().numpy())
    #         pred_lis += batch_pred
    #         gt_lis += list(labels.detach().cpu().numpy())

    #         top5_pred = torch.topk(logits, 5, dim=-1)[1].detach().cpu().numpy()
    #         top5_pred_lis += list(top5_pred)

            
    #         ind_logits += list(max_logit_score(logits))
    #         ind_prob += list(msp_score(logits))
    #         ind_energy += list(energy_score(logits))
    #         def top_k_accuracy(gt, pred, k=5):
    #             correct = 0
    #             for i in range(len(gt)):
    #                 if gt[i] in pred[i][:k]:
    #                     correct += 1
    #             return correct / len(gt)
            
    #         if flag:
    #             idex = torch.argmax(logits, -1).unsqueeze(-1)
    #             yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
    #             yesno = torch.softmax(yesno, dim=-1)[:,:,0]
    #             yesno_s = torch.gather(yesno, dim=1, index=idex)
    #             ind_ctw += list(yesno_s.detach().cpu().numpy())
    #             batch_ind_atd = list((yesno * torch.softmax(logits, -1)).sum(1).detach().cpu().numpy())
    #             ind_atd += batch_ind_atd
                

                
            
        # for name, ood_data in ood_dataset.items():
        #     ood_logits, ood_prob, ood_energy = [], [], []
        #     if flag:
        #         ood_ctw, ood_atd = [], []
        #     for i, batch in tqdm(enumerate(ood_data)):
        #         batch = maybe_dictionarize(batch)
        #         inputs = batch["images"].cuda()
        #         labels = batch['labels'].cuda()
        #         logits, logits_no, _ = model(inputs)
                
        #         ood_logits += list(max_logit_score(logits))
        #         ood_prob += list(msp_score(logits))
        #         ood_energy += list(energy_score(logits))
            
        #         if flag:
        #             idex = torch.argmax(logits, -1).unsqueeze(-1) #j
        #             yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
        #             yesno = torch.softmax(yesno, dim=-1)[:,:,0]
        #             yesno_s = torch.gather(yesno, dim=1, index=idex)

        #             ood_ctw += list(yesno_s.detach().cpu().numpy())
        #             batch_ood_atd = list((yesno * torch.softmax(logits, -1) ).sum(1).detach().cpu().numpy())
        #             ood_atd += batch_ood_atd
                    
                    
                 
            #### MSP
            auc, fpr = cal_auc_fpr(ind_prob, ood_prob)
            res.append([epoch, "MSP", name, auc, fpr])
            #### MaxLogit
            auc, fpr = cal_auc_fpr(ind_logits, ood_logits)
            res.append([epoch, "MaxLogit", name, auc, fpr])
            #### Energy
            auc, fpr = cal_auc_fpr(ind_energy, ood_energy)
            res.append([epoch, "Energy", name, auc, fpr])
            if flag:
                auc, fpr = cal_auc_fpr(ind_ctw, ood_ctw)
                res.append([epoch, "CTW", name, auc, fpr])
                
                auc, fpr = cal_auc_fpr(ind_atd, ood_atd)
                res.append([epoch, "ATD", name, auc, fpr])
                
            
    pred_lis = np.array(pred_lis)
    top5_pred_lis = np.array(top5_pred_lis)
    gt_lis = np.array(gt_lis)
    acc = Acc(gt_lis, pred_lis)
    top_5_acc = top_k_accuracy(gt_lis, top5_pred_lis, k=5)
    
    id_lis_epoch = [[epoch, acc]]
    ood_lis_epoch = res
    print(id_lis_epoch)
    print(f"Top-5 Accuracy: {top_5_acc * 100:.2f}%")
    for lis in ood_lis_epoch:
        print(lis)
    import json
    for entry in json_data:
        # Convert bbox if it's a tensor to a list of floats
        if torch.is_tensor(entry['bbox']):
            entry['bbox'] = entry['bbox'].tolist()
        
        # Convert score if it's a tensor to a float
        if torch.is_tensor(entry['score']):
            entry['score'] = entry['score'].item()
        
        # Convert category_id if it's a tensor to an integer
        if torch.is_tensor(entry['category_id']):
            entry['category_id'] = entry['category_id'].item()

    for entry in json_data:
        # Convert bbox if it's a NumPy array or a tensor to a list of floats
        if isinstance(entry['bbox'], np.ndarray):
            entry['bbox'] = entry['bbox'].tolist()
        elif torch.is_tensor(entry['bbox']):
            entry['bbox'] = entry['bbox'].tolist()
        
        # Convert score if it's a NumPy array or a tensor to a float
        if isinstance(entry['score'], np.ndarray):
            entry['score'] = entry['score'].item()  # Assuming score is a scalar, you might use .item() instead if appropriate
        elif torch.is_tensor(entry['score']):
            entry['score'] = entry['score'].item()
        
        # Convert category_id if it's a NumPy array or a tensor to an integer
        if isinstance(entry['category_id'], np.ndarray):
            entry['category_id'] = int(entry['category_id'])
        elif torch.is_tensor(entry['category_id']):
            entry['category_id'] = entry['category_id'].item()

    # Serialize to JSON

    target_path = os.path.join(pth_dir, "vis.json")
    with open(target_path, 'w') as f:
        json.dump(json_data, f,indent=4)

    return id_lis_epoch, ood_lis_epoch

def visualize(pth_dir,id_dataset, backbone,rpn,vit_class, target_folder_name,ood_dataset=None, flag = True):
    backbone.eval()
    rpn.eval()
    vit_class.eval()
    if flag:
        ind_ctw, ind_atd = [], []
    
    # three lists for storing values for visualization
    # classes = [] # top1 predicted class id
    # id_scores = []
    # ood_scores = []
    classes = []
    ind_atd_values = []
    ood_atd_values = []


    with torch.no_grad():
        for i, batch in tqdm(enumerate(id_dataset)):
            images, bboxes_info = batch
            # images, bboxes_info = batch['image'],batch['bboxes_info']
            #images: torch.Size([512, 3, 224, 224])
            bboxes=bboxes_info['bboxes'] #tensor list of coordinates: (512,k,4)
            texts=bboxes_info['texts'] # list, (512,k)
            bbox_labels=bboxes_info['bbox_labels'] # list, (512,k)
            original_images=bboxes_info['original_images'] # list,(512,PIL with original size)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            images = images.to(device=device, non_blocking=True)
            # texts = texts.to(device=device, non_blocking=True)
            bbox_targets=[]
            
            # for labels, boxes in zip(bbox_labels, bboxes[0]):
            #     target_dict = {
            #         'boxes': boxes,  # 如果有多个框，应该是torch.stack(boxes)
            #         'labels': torch.tensor(labels)
            #     }
            #     bbox_targets.append(target_dict)
            for labels, boxes in zip(bbox_labels, bboxes):
                target_dict = {
                    'boxes': boxes,
                    'labels': torch.tensor(labels)
                }
                bbox_targets.append(target_dict)

            #forward
            bbox_targets = [{k: v.to(device) for k, v in t.items()} for t in bbox_targets]
            backbone_features=backbone(images)
            height, width = images.shape[2], images.shape[3]
            image_sizes = [(height, width)] * images.shape[0] 
            images_list = ImageList(images, image_sizes)
            # backbone_features = F.normalize(backbone_features, dim=-1)
            proposals, proposal_losses = rpn(images_list, backbone_features, targets=bbox_targets)
            valid_cropped_images,valid_labels,gt_cropped_images,gt_labels,valid_proposals=get_valid_cropped_images_texts(images,proposals,original_images,bbox_targets,IoU_threshold=0.5)
            # resize_transform = transforms.Resize((224, 224))
            # valid_cropped_images= [resize_transform(img) for img in valid_cropped_images]
            valid_cropped_images = torch.stack(valid_cropped_images)
            valid_cropped_images=valid_cropped_images.to(device)
            # gt_cropped_images=[resize_transform(img) for img in gt_cropped_images]
            # gt_cropped_images=torch.stack(gt_cropped_images)
            # gt_cropped_images=gt_cropped_images.to(device)
            
            
            valid_descriptions = labels_to_descriptions_COCO(valid_labels)
            from open_clip import tokenize
            valid_texts=[tokenize(text) for text in valid_descriptions]
            valid_texts=torch.stack(valid_texts).to(device)
            
            # gt_descriptions = labels_to_descriptions_COCO(gt_labels)
            # from open_clip import tokenize
            # gt_texts=[tokenize(text) for text in gt_descriptions]
            # gt_texts=torch.stack(gt_texts).to(device)
            
            
            # valid_cropped_images=torch.cat((valid_cropped_images, gt_cropped_images), dim=0)
            # valid_texts=torch.cat((valid_texts, gt_texts), dim=0)
            
            # image_features, text_features, text_features_no,logit_scale=self.clip_model(valid_cropped_images,valid_texts)
            logits, logits_no, _=vit_class(valid_cropped_images)
            
            batch_pred = list(torch.argmax(logits, -1).detach().cpu().numpy())
            classes.extend(batch_pred)
            
            if flag:
                idex = torch.argmax(logits, -1).unsqueeze(-1)
                yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                yesno = torch.softmax(yesno, dim=-1)[:,:,0]
                yesno_s = torch.gather(yesno, dim=1, index=idex)
                batch_ind_ctw = list(yesno_s.detach().cpu().numpy())
                ind_ctw += batch_ind_ctw
                batch_ind_atd = list((yesno * torch.softmax(logits, -1)).sum(1).detach().cpu().numpy())
                ind_atd += batch_ind_atd
                ind_atd_values.extend(batch_ind_ctw)    
            
        for name, ood_data in ood_dataset.items():
            ood_logits, ood_prob, ood_energy = [], [], []
            if flag:
                ood_ctw, ood_atd = [], []
            for i, batch in tqdm(enumerate(ood_data)):
                images, bboxes_info = batch
                # images, bboxes_info = batch['image'],batch['bboxes_info']
                #images: torch.Size([512, 3, 224, 224])
                bboxes=bboxes_info['bboxes'] #tensor list of coordinates: (512,k,4)
                texts=bboxes_info['texts'] # list, (512,k)
                bbox_labels=bboxes_info['bbox_labels'] # list, (512,k)
                original_images=bboxes_info['original_images'] # list,(512,PIL with original size)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                images = images.to(device=device, non_blocking=True)
                # texts = texts.to(device=device, non_blocking=True)
                bbox_targets=[]
                
                # for labels, boxes in zip(bbox_labels, bboxes[0]):
                #     target_dict = {
                #         'boxes': boxes,  # 如果有多个框，应该是torch.stack(boxes)
                #         'labels': torch.tensor(labels)
                #     }
                #     bbox_targets.append(target_dict)
                for labels, boxes in zip(bbox_labels, bboxes):
                    target_dict = {
                        'boxes': boxes,
                        'labels': torch.tensor(labels)
                    }
                    bbox_targets.append(target_dict)

                #forward
                bbox_targets = [{k: v.to(device) for k, v in t.items()} for t in bbox_targets]
                backbone_features=backbone(images)
                height, width = images.shape[2], images.shape[3]
                image_sizes = [(height, width)] * images.shape[0] 
                images_list = ImageList(images, image_sizes)
                # backbone_features = F.normalize(backbone_features, dim=-1)
                proposals, proposal_losses = rpn(images_list, backbone_features, targets=bbox_targets)
                valid_cropped_images,valid_labels,gt_cropped_images,gt_labels,valid_proposals=get_valid_cropped_images_texts(images,proposals,original_images,bbox_targets,IoU_threshold=0.5)
                # resize_transform = transforms.Resize((224, 224))
                # valid_cropped_images= [resize_transform(img) for img in valid_cropped_images]
                valid_cropped_images = torch.stack(valid_cropped_images)
                valid_cropped_images=valid_cropped_images.to(device)
                # gt_cropped_images=[resize_transform(img) for img in gt_cropped_images]
                # gt_cropped_images=torch.stack(gt_cropped_images)
                # gt_cropped_images=gt_cropped_images.to(device)
                
                
                valid_descriptions = labels_to_descriptions_COCO(valid_labels)
                from open_clip import tokenize
                valid_texts=[tokenize(text) for text in valid_descriptions]
                valid_texts=torch.stack(valid_texts).to(device)
                
                # gt_descriptions = labels_to_descriptions_COCO(gt_labels)
                # from open_clip import tokenize
                # gt_texts=[tokenize(text) for text in gt_descriptions]
                # gt_texts=torch.stack(gt_texts).to(device)
                
                
                # valid_cropped_images=torch.cat((valid_cropped_images, gt_cropped_images), dim=0)
                # valid_texts=torch.cat((valid_texts, gt_texts), dim=0)
                
                # image_features, text_features, text_features_no,logit_scale=self.clip_model(valid_cropped_images,valid_texts)
                logits, logits_no, _=vit_class(valid_cropped_images)
                
                ood_logits += list(max_logit_score(logits))
                ood_prob += list(msp_score(logits))
                ood_energy += list(energy_score(logits))
            
                if flag:
                    idex = torch.argmax(logits, -1).unsqueeze(-1) #j
                    yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                    yesno = torch.softmax(yesno, dim=-1)[:,:,0]
                    yesno_s = torch.gather(yesno, dim=1, index=idex)

                    batch_ood_ctw = list(yesno_s.detach().cpu().numpy())
                    ood_ctw +=batch_ood_ctw
                    batch_ood_atd = list((yesno * torch.softmax(logits, -1) ).sum(1).detach().cpu().numpy())
                    ood_atd += batch_ood_atd
                    ood_atd_values.extend(batch_ood_ctw)
    
    
    # classes
    # id_scores
    # ood_scores
    scores_json = {
    'classes':classes,
    'id_scores': ind_atd_values,
    'ood_scores': ood_atd_values
}
    # fix serizliation issue
    # print(type(scores_json['classes']))
    # print(type(scores_json['classes'][0]))
    # print(type(scores_json['id_scores']))
    # print(type(scores_json['id_scores'][0]))
    # print(type(scores_json['ood_scores']))
    # print(type(scores_json['ood_scores'][0]))
    # print('after')
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):  
            return [convert_numpy(item) for item in obj] 
        else:
            return obj

    # Apply the conversion function to each item in the dictionary
    scores_json_converted = {k: convert_numpy(v) for k, v in scores_json.items()}
    # print(type(scores_json_converted['classes']))
    # print(type(scores_json_converted['classes'][0]))
    # print(type(scores_json_converted['id_scores']))
    # print(type(scores_json_converted['id_scores'][0]))
    # print(type(scores_json_converted['ood_scores']))
    # print(type(scores_json_converted['ood_scores'][0]))
    # Write to file

    import json
    target_path = os.path.join(target_folder_name, "vis.json")
    with open(target_path, 'w') as f:
        json.dump(scores_json_converted, f)

def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

def cal_fpr_recall(ind_conf, ood_conf, tpr=0.95):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return fpr, thresh
    



if __name__ == '__main__':
    args = parse_arguments()
    
    pth_dir = '/home/ywan1084/Documents/Github/cloud/src/logs/2024_05_03-14_41_02-model_ViT-B-16-lr_0.0003-b_32-j_1-p_amp'

    header_ood = ['epoch', 'method', 'oodset', 'AUROC', 'FPR@95']
    ood_lis = []
    if "ViT-B-16" in pth_dir:
        model_type = "ViT-B-16"
    elif "ViT-B-32" in pth_dir:
        model_type = "ViT-B-32"
    elif "ViT-L-14" in pth_dir:
        model_type = "ViT-L-14"
    start,end = 10,11
    for i in range(start,end):    ### evaluate the model of the 10-th epoch.
        if i==end-1:
            ood_lis += infer(args, pth_dir, i, model_type=model_type,vis=False)
        else: 
            ood_lis += infer(args, pth_dir, i, model_type=model_type,vis=False)

    df = pd.DataFrame(ood_lis, columns=header_ood)

    df.to_csv(os.path.join(pth_dir, 'ood_metric_.csv'), index=False)