import yaml
import setproctitle

from dingo.gw.dataset import generate_dataset

setproctitle.setproctitle("GenerateWfs")
out_file = "/data/nihargupte/datasets/waveforms/aligned_spin_SEOBNRv4HM_PA_2048.hdf5"
with open("/home/local/nihargupte/dingo-devel/tutorials/06_tgr/datasets/waveforms/settings_SEOBNRv4HM_PA_FD.yaml", "r") as f:
    settings = yaml.safe_load(f)

dataset = generate_dataset(settings, num_processes=32)
dataset.to_file(out_file)