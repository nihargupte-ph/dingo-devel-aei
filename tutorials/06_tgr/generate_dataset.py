import yaml
import setproctitle

from dingo.gw.dataset import generate_dataset

setproctitle.setproctitle("GenerateWfs")
out_file = "/data/nihargupte/datasets/waveforms/aligned_spin_SEOBNRv4HM_PA_2048_O3.hdf5"
with open("/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/datasets/waveforms/settings_SEOBNRv4HM_PA_O3_priors.yaml","r") as f:
    settings = yaml.safe_load(f)

dataset = generate_dataset(settings, num_processes=16)
dataset.to_file(out_file)
