"""
 Data related (creation, preprocessing, paths...) parameters
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - June 2024
"""

# Paths
train_path = r"data/split1/train/"
val_path = r"data/split1/val/"
test_path = r"data/split1/test/"
visu_path = r"data/split1/visu/"
save_path = r"results/tmp/dummy_experiment/"

# Data parameters
list_N = [100,100,100] # Number of points per class
list_centroid = [[-1,-1],[1,1],[0.5,-0.5]]