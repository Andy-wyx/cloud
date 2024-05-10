import torch

# 加载模型
# model1 = torch.load("/data/kangyuzhu/ViT-B-16/checkpoints/epoch_10.pt")
model = torch.load("/home/ywan1084/Documents/Github/cloud/src/logs/2024_05_03-14_41_02-model_ViT-B-16-lr_0.0003-b_32-j_1-p_amp/checkpoints/epoch_10.pt")

# # named_modules = model.named_modules()## 获取所有模块的名称
# state_dict1 = model1['state_dict']
state_dict = model['state_dict'] # for clipn
rpn = model['rpn']
backbone = model['backbone']
# # 找出两个状态字典中不同的键
# diff_keys = set(state_dict1.keys()) ^ set(state_dict2.keys())

# # 打印不同层的名称
# print("Different layers between model1 and model2:")
for key in state_dict.keys():
    print(key)
print("")
for key in rpn.keys():
    print(key)
print("")
for key in backbone.keys():
    print(key) 

# 获取模型1和模型2的层名
# model1_layers = set(model1.keys())
# model2_layers = set(model2.keys())

# 打印出模型1有的但模型2没有的层
# print("Layers in model1 but not in model2:")
# print(model1_layers - model2_layers)

# # 打印出模型2有的但模型1没有的层
# print("Layers in model2 but not in model1:")
# print(model2_layers - model1_layers)


# import torch

# # 加载模型
# model = torch.load("/home/kangyuzhu/CLIPN/src/logs/2024_04_29-20_10_42-model_ViT-B-16-lr_0.0003-b_128-j_4-p_amp/checkpoints/epoch_10.pt")

# # # 新建一个字典，用于保存修改后的权重
# new_state_dict = {}
# state_dict = model['state_dict']
# # # 遍历模型的状态字典，修改键并保存权重
# # print(state_dict.items())
# for name, param in state_dict.items():
#     # print(name)
#     if name.startswith("module.clip_model."):
#         # 替换键中的 "module.clip_model." 为 "module."
        
#         new_name = name.replace("module.clip_model.", "module.")
#         # 保存修改后的权重
#         # print(new_name)
#         new_state_dict[new_name] = param
#     else:
#         # 对于不含 "module.clip_model." 前缀的键，直接保存到新的字典中
#         new_state_dict[name] = param
#     # break

# # # 将修改后的状态字典保存到原来的文件路径
# torch.save({
#     "state_dict":new_state_dict
#     }, 
#            "/home/kangyuzhu/CLIPN/src/logs/2024_04_29-20_10_42-model_ViT-B-16-lr_0.0003-b_128-j_4-p_amp/checkpoints/model_10_new.pt")