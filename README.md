# OriginQ-wukong-playground

    Run basic experiments on the OriginQ "wukong" real quantum computer

----

2024年1月12日 本源量子量子计算真机 “本源悟空” [发布](https://originqc.com.cn/zh/new_detail.html?newId=396)  
注册就送 10min (价值1w RMB) 的使用时长，可惜 2024.2.6 就过期了，让我们来跑几个经典实验吧！ ;)  

⚪ without optim

| name | results | time usage (sec) | 
| :-: | :-: | :-: |
| coin_simple | {'0': 0.521, '1': 0.479} | 0.508 |
| coin_dangling | {'00': 0.481, '01': 0.456, '10': 0.036, '11': 0.027} | 0.523 |

⚪ with optim

| name | results | time usage (sec) | 
| :-: | :-: | :-: |
| coin | { "0": 0.4630641862546881, "1": 0.5369358137453119 } | 2.779 |
| bell-state | {"00": 0.48219576362296784, "01": 8.731294366183374e-05, "10": 0.004007638460295128, "11": 0.513709284973075 } | 2.923 |
| triple-dice | {'00': 0.395, '01': 0.313, '10': 0.27, '11': 0.022} | 3.038 |
| swap | {'00': 0.6238352427724224, '01': 0.0008265783363700312, '10': 0.3717188442764467, '11': 0.003619334614761154} | 2.616 |

----
by Armit
2024/02/05