# Loading the data that we got from the LALInference run instead of just specifying the GPS time
from gwpy.timeseries import TimeSeries
from lalframe.frread import read_timeseries
import lal 
import lalframe
import os 

# Copied directly from LALinference run except had to move where the file was since saraswati can't access HYPATIA localhost

source = "/home/nihargupte.HYPATIA/data/frame_files/GW150914/L-L1_HOFT_C02_CACHE-1126259184-297.lcf"
source = lal.CacheImport(os.path.expanduser(source))
lalframe.FrCacheOpen(source)
# print(dir(source))
L1_data = read_timeseries(source=source, channel="L1:DCS-CALIB_STRAIN_C02")
# lalinference_strain_data_L1 = TimeSeries.read(source=source, channel="L1:DCS-CALIB_STRAIN_C02")
# lalinference_strain_data_L1 = TimeSeries.read("/home/nihargupte.HYPATIA/data/frame_files/GW150914/L-L1_HOFT_C02-1126256640-4096.gwf", "L1:DCS-CALIB_STRAIN_C02")
# lalinference_strain_data_H1 = TimeSeries.read("/home/nihargupte.HYPATIA/data/frame_files/GW150914/H-H1_HOFT_C02-1126256640-4096.gwf", "H1:DCS-CALIB_STRAIN_C02")
