import yaml
from dingo.gw.training.train_pipeline import *
import setproctitle
import torch
from pprint import pprint

torch.cuda.set_device(3)
setproctitle.setproctitle("dingo_train_O3_time")

train_dir_name = "train_dir_SEOBNRv4HM_PA_O3_1600_HLV_time"
train_settings_file = f"/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/{train_dir_name}/train_settings.yaml"
train_dir = f"/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/{train_dir_name}/"
checkpoint = f"/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/{train_dir_name}/model_latest.pt"
with open(train_settings_file, "r") as fp:
    train_settings = yaml.safe_load(fp)

# Extract the local settings from train settings file, save it separately. This
# file can later be modified, and the settings take effect immediately upon
# resuming.

local_settings = train_settings.pop("local")
with open(os.path.join(train_dir, "local_settings.yaml"), "w") as f:
    yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

# pm, wfd = prepare_training_new(train_settings, train_dir, local_settings)
pm, wfd = prepare_training_resume(checkpoint, local_settings, train_dir)

print("got here")
with threadpool_limits(limits=1, user_api="blas"):
    complete = train_stages(pm, wfd, train_dir, local_settings)
