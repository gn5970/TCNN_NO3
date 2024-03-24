import numpy as np
import matplotlib.pyplot as plt
#import CMIP6_second_GFDL_atlantic

skill_atlantic=np.zeros((9))
auto_atlantic=np.zeros((98))

#import CMIP6_second_GFDL_atlantic
skill_pacific=np.zeros((9))
auto_pacific=np.zeros((98))

skill_indian=np.zeros((9))
auto_indian=np.zeros((98))

skill_so=np.zeros((9))
auto_so=np.zeros((98))

 
exec(open("final_code_no3_with_reg.py").read())

#import CMIP6_second_GFDL_pacific
#exec(open("final_code_npp.py").read())

#exec(open("final_code_chl.py").read())

exec(open("final_code_NO3_Network_baseline2.py").read())
#import CMIP6_second_GFDL_indian
#exec(open("CMIP6_second_GFDL_composite_pacific_ig.py").read())

#import CMIP6_second_GFDL
#exec(open("CMIP6_second_GFDL_composite_atlantic_ig.py").read())
