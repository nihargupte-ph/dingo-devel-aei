from dingo.gw.ASD_dataset.generate_dataset import generate_dataset
import yaml
import setproctitle


setproctitle.setproctitle("GenerateASDsO3")
# Testing my function for generating PSDs from a GWF file
data_dir = "/data/nihargupte/datasets/ASDs/O3/"
with open(data_dir + "settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

generate_dataset(data_dir=data_dir, settings=settings)